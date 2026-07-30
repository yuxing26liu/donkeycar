#!/usr/bin/env python3
"""

Scripts to drive on autopilot using computer vision

Usage:
    manage.py (drive) [--js] [--log=INFO] [--camera=(single|stereo)] [--myconfig=<filename>] [--meta=<key:value> ...]


Options:
    -h --help          Show this screen.
    --js               Use physical joystick.
    --myconfig=filename     Specify myconfig file to use.
                            [default: myconfig.py]
    --meta=<key:value>      Key/value strings describing a piece of meta
                            data about this drive (e.g. lane:right). Option
                            may be used more than once.
"""
import logging

from docopt import docopt
from simple_pid import PID

import donkeycar as dk
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.line_follower import LineFollower
from donkeycar.templates.complete import add_odometry, add_camera, \
    add_user_controller, add_drivetrain, add_simulator, add_imu, DriveMode, \
    UserPilotCondition, ToggleRecording
from donkeycar.parts.logger import LoggerPart
from donkeycar.parts.transform import Lambda
from donkeycar.parts.explode import ExplodeDict
from donkeycar.parts.controller import JoystickController

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def drive(cfg, use_joystick=False, camera_type='single', meta=[]):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''
    
    #Initialize car
    V = dk.vehicle.Vehicle()

    #
    # if we are using the simulator, set it up
    #
    add_simulator(V, cfg)

    #
    # setup primary camera
    #
    add_camera(V, cfg, camera_type)

    #
    # add the user input controller(s)
    # - this will add the web controller
    # - it will optionally add any configured 'joystick' controller
    #
    has_input_controller = hasattr(cfg, "CONTROLLER_TYPE") and cfg.CONTROLLER_TYPE != "mock"
    ctr = add_user_controller(V, cfg, use_joystick, input_image = 'ui/image_array')

    #
    # explode the web buttons into their own key/values in memory
    #
    V.add(ExplodeDict(V.mem, "web/"), inputs=['web/buttons'])

    #
    # track user vs autopilot condition
    #
    V.add(UserPilotCondition(show_pilot_image=getattr(cfg, 'OVERLAY_IMAGE', False)),
          inputs=['user/mode', "cam/image_array", "cv/image_array"],
          outputs=['run_user', "run_pilot", "ui/image_array"])

    #
    # PID controller to be used with cv_controller
    #
    pid = PID(Kp=cfg.PID_P, Ki=cfg.PID_I, Kd=cfg.PID_D)
    def dec_pid_d():
        pid.Kd -= cfg.PID_D_DELTA
        logging.info("pid: d- %f" % pid.Kd)

    def inc_pid_d():
        pid.Kd += cfg.PID_D_DELTA
        logging.info("pid: d+ %f" % pid.Kd)

    def dec_pid_p():
        pid.Kp -= cfg.PID_P_DELTA
        logging.info("pid: p- %f" % pid.Kp)

    def inc_pid_p():
        pid.Kp += cfg.PID_P_DELTA
        logging.info("pid: p+ %f" % pid.Kp)

    #
    # Computer Vision Controller
    #
    cv_controller = add_cv_controller(V, cfg, pid,
                      cfg.CV_CONTROLLER_MODULE,
                      cfg.CV_CONTROLLER_CLASS,
                      cfg.CV_CONTROLLER_INPUTS,
                      cfg.CV_CONTROLLER_OUTPUTS,
                      cfg.CV_CONTROLLER_CONDITION)

    #
    # Cone-avoidance planner (cone-dodger3000 branch) - disabled unless
    # OBSTACLE_AVOIDANCE_MODE is set to 'observe'/'shadow'/'active' in
    # myconfig.py (default 'disabled': the part below isn't even added,
    # so a car without this configured pays zero cost for it). Requires
    # CV_CONTROLLER_CLASS to be LaneFollower (needs .set_lane()/
    # .current_lane/.white_right_of_yellow/.scan_rows) - silently skipped
    # otherwise, e.g. when running the baseline LineFollower.
    # 'active' mode is the only mode that ever calls set_lane() or
    # changes what actually drives the car - see pilot_arbiter.py.
    obstacle_avoidance_mode = getattr(cfg, 'OBSTACLE_AVOIDANCE_MODE', 'disabled')
    if obstacle_avoidance_mode != 'disabled' and cfg.CV_CONTROLLER_CLASS == 'LaneFollower':
        from donkeycar.parts.cone_detector import ConeDetector
        from donkeycar.parts.obstacle_planner import ObstaclePlanner
        from donkeycar.parts.pilot_arbiter import PilotArbiter

        cone_detector = ConeDetector(cfg)
        obstacle_planner = ObstaclePlanner(cfg)
        pilot_arbiter = PilotArbiter(cfg, cv_controller, cone_detector, obstacle_planner)
        V.add(pilot_arbiter,
              inputs=['cam/image_array', 'cam/depth_array', 'pilot/steering', 'pilot/throttle',
                      'lane/yellow_x', 'lane/white_x', 'lane/width_px'],
              outputs=['pilot/steering', 'pilot/throttle',
                       'planner/state', 'planner/reason', 'planner/requested_lane',
                       'planner/cone_in_path', 'cone/bbox_x', 'cone/bbox_y', 'cone/bbox_w',
                       'cone/bbox_h', 'cone/distance_mm', 'cone/distance_valid',
                       'cone/distance_source'],
              run_condition=cfg.CV_CONTROLLER_CONDITION)
        logger.info(f"Cone avoidance wired in: OBSTACLE_AVOIDANCE_MODE={obstacle_avoidance_mode!r} "
                    f"(only 'active' calls set_lane() or changes throttle)")
    elif obstacle_avoidance_mode != 'disabled':
        logger.warning(f"OBSTACLE_AVOIDANCE_MODE={obstacle_avoidance_mode!r} but CV_CONTROLLER_CLASS="
                        f"{cfg.CV_CONTROLLER_CLASS!r} (not LaneFollower) -- cone avoidance not wired in.")

    recording_control = ToggleRecording(cfg.AUTO_RECORD_ON_THROTTLE, cfg.RECORD_DURING_AI)
    V.add(recording_control, inputs=['user/mode', "recording"], outputs=["recording"])


    #
    # Add buttons for handling various user actions
    # The button names are in configuration.
    # They may refer to game controller (joystick) buttons OR web ui buttons
    #
    # There are 5 programmable webui buttons, "web/w1" to "web/w5"
    # adding a button handler for a webui button
    # is just adding a part with a run_condition set to
    # the button's name, so it runs when button is pressed.
    #
    have_joystick = ctr is not None and isinstance(ctr, JoystickController)

    # button to toggle recording
    if cfg.TOGGLE_RECORDING_BTN:
        print(f"Toggle recording button is {cfg.TOGGLE_RECORDING_BTN}")
        if cfg.TOGGLE_RECORDING_BTN.startswith("web/w"):
            V.add(Lambda(lambda: recording_control.toggle_recording()), run_condition=cfg.TOGGLE_RECORDING_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.TOGGLE_RECORDING_BTN, recording_control.toggle_recording)

    # Buttons to tune PID constants
    if cfg.DEC_PID_P_BTN and cfg.PID_P_DELTA:
        print(f"Decrement PID P button is {cfg.DEC_PID_P_BTN}")
        if cfg.DEC_PID_P_BTN.startswith("web/w"):
            V.add(Lambda(lambda: dec_pid_p()), run_condition=cfg.DEC_PID_P_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.DEC_PID_P_BTN, dec_pid_p)
    if cfg.INC_PID_P_BTN and cfg.PID_P_DELTA:
        print(f"Increment PID P button is {cfg.INC_PID_P_BTN}")
        if cfg.INC_PID_P_BTN.startswith("web/w"):
            V.add(Lambda(lambda: inc_pid_p()), run_condition=cfg.INC_PID_P_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.INC_PID_P_BTN, inc_pid_p)
    if cfg.DEC_PID_D_BTN and cfg.PID_D_DELTA:
        print(f"Decrement PID D button is {cfg.DEC_PID_D_BTN}")
        if cfg.DEC_PID_D_BTN.startswith("web/w"):
            V.add(Lambda(lambda: dec_pid_d()), run_condition=cfg.DEC_PID_D_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.DEC_PID_D_BTN, dec_pid_d)
    if cfg.INC_PID_D_BTN and cfg.PID_D_DELTA:
        print(f"Increment PID D button is {cfg.INC_PID_D_BTN}")
        if cfg.INC_PID_D_BTN.startswith("web/w"):
            V.add(Lambda(lambda: inc_pid_d()), run_condition=cfg.INC_PID_D_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.INC_PID_D_BTN, inc_pid_d)

    #
    # Decide what inputs should change the car's steering and throttle
    # based on the choice of user or autopilot drive mode
    #
    V.add(DriveMode(cfg.AI_THROTTLE_MULT),
          inputs=['user/mode', 'user/steering', 'user/throttle',
                  'pilot/steering', 'pilot/throttle'],
          outputs=['steering', 'throttle'])


    #
    # Setup drivetrain
    #
    add_drivetrain(V, cfg)


    #
    # OLED display setup
    #
    if cfg.USE_SSD1306_128_32:
        from donkeycar.parts.oled import OLEDPart
        auto_record_on_throttle = cfg.USE_JOYSTICK_AS_DEFAULT and cfg.AUTO_RECORD_ON_THROTTLE
        oled_part = OLEDPart(cfg.SSD1306_128_32_I2C_ROTATION, cfg.SSD1306_RESOLUTION, auto_record_on_throttle)
        V.add(oled_part, inputs=['recording', 'tub/num_records', 'user/mode'], outputs=[], threaded=True)


    #
    # add tub to save data
    #
    # lane/yellow_x, lane/white_x, lane/width_px are only ever populated
    # when CV_CONTROLLER_CLASS is LaneFollower (see lane_follower.py) -
    # Memory.get() returns None for a key nothing has written, and
    # Tub.write_record() skips None values, so this is a no-op (those
    # three columns just never appear) when running LineFollower instead.
    # Recording them lets a tub be analyzed for exactly what was detected
    # per frame, instead of having to infer it from steering/throttle.
    #
    # cam/depth_array is likewise only populated when CAMERA_TYPE="OAKD"
    # (or "D435") and *_DEPTH is enabled (see add_camera() in complete.py) -
    # same None-skip no-op otherwise. Recorded as a 16-bit PNG (gray16_array)
    # so raw depth can be analyzed offline alongside the RGB frame.
    #
    # planner/* and cone/* are only populated when OBSTACLE_AVOIDANCE_MODE
    # is not 'disabled' (see pilot_arbiter.py) - same None-skip no-op
    # otherwise. Recording these is what makes an on-car cone-avoidance
    # test tub analyzable offline frame-by-frame (state/reason per frame,
    # not just inferred from steering/throttle after the fact).
    inputs=['cam/image_array',
            'steering', 'throttle',
            'lane/yellow_x', 'lane/white_x', 'lane/width_px',
            'cam/depth_array',
            'planner/state', 'planner/reason', 'planner/requested_lane', 'planner/cone_in_path',
            'cone/bbox_x', 'cone/bbox_y', 'cone/bbox_w', 'cone/bbox_h',
            'cone/distance_mm', 'cone/distance_valid', 'cone/distance_source']

    types=['image_array',
           'float', 'float',
           'float', 'float', 'float',
           'gray16_array',
           'str', 'str', 'str', 'boolean',
           'float', 'float', 'float', 'float',
           'float', 'boolean', 'str']

    #
    # Create data storage part
    #
    tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path() if \
        cfg.AUTO_CREATE_NEW_TUB else cfg.DATA_PATH
    meta += getattr(cfg, 'METADATA', [])
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    if cfg.DONKEY_GYM:
        print("You can now go to http://localhost:%d to drive your car." % cfg.WEB_CONTROL_PORT)
    else:
        print("You can now go to <your hostname.local>:%d to drive your car." % cfg.WEB_CONTROL_PORT)
    if has_input_controller:
        print("You can now move your controller to drive your car.")
        if isinstance(ctr, JoystickController):
            ctr.set_tub(tub_writer.tub)
            ctr.print_controls()

    #
    # run the vehicle
    #
    # verbose=True (set VERBOSE_CAR in myconfig.py) gets you two things
    # Vehicle.start() otherwise stays silent about: a WARN log line any
    # loop that can't keep up with DRIVE_LOOP_HZ, and a per-part timing
    # table (max/min/avg/percentiles, in ms) printed every 200 loops -
    # the direct way to see how long LaneFollower's run() actually takes
    # per frame, rather than inferring it from tub timestamps after the
    # fact.
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS,
            verbose=getattr(cfg, 'VERBOSE_CAR', False))


#
# Computer Vision Controller
#
def add_cv_controller(
        V, cfg, pid,
        module_name="donkeycar.parts.line_follower",
        class_name="LineFollower",
        inputs=['cam/image_array'],
        outputs=['pilot/steering', 'pilot/throttle', 'cv/image_array'],
        run_condition="run_pilot"):

        # __import__ the module
        module = __import__(module_name)

        # walk module path to get to module with class
        for attr in module_name.split('.')[1:]:
            module = getattr(module, attr)

        my_class = getattr(module, class_name)

        # add instance of class to vehicle
        controller = my_class(pid, cfg)
        V.add(controller,
              inputs=inputs,
              outputs=outputs,
              run_condition=run_condition)
        # returned so drive() can hand the SAME instance to PilotArbiter
        # (needs .set_lane()/.current_lane, not just its Memory outputs)
        return controller


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    log_level = args['--log'] or "INFO"
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=numeric_level)

    if args['drive']:
        drive(cfg, use_joystick=args['--js'], camera_type=args['--camera'],
              meta=args['--meta'])
