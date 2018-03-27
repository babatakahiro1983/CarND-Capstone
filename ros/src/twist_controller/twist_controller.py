from pid import PID


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, max_yaw_rate, accel_limit, decel_limit):
        self.max_yaw_rate = max_yaw_rate
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit

        steer_kp = 0.2
        steer_ki = 0.01
        steer_kd = 0.001

        self.steer_pid = PID(steer_kp,
                             steer_ki,
                             steer_kd,
                             0. - max_yaw_rate,
                             max_yaw_rate)
        
        throttle_kp = 1
        throttle_ki = 0
        throttle_kd = 0
        
        self.throttle_pid = PID(throttle_kp,
                                throttle_ki,
                                throttle_kd,
                                decel_limit,
                                accel_limit)

    def control(self, target_acceleration, target_yaw_rate, deltat, dbw_enabled):
        acceleration = self.throttle_pid.step(target_acceleration, deltat)
        yaw_rate = self.steer_pid.step(target_yaw_rate, deltat)

        if not dbw_enabled:
            self.throttle_pid.reset()
            self.steer_pid.reset()

        throttle = acceleration
        steer = yaw_rate * 14.8
        brake = 0
        if throttle < 0.:
            brake = 0. - throttle
            throttle = 0.

        return throttle, brake, steer

    def reset_throttle_pid(self):
        self.throttle_pid.reset()
