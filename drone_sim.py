import math
import pygame
from dataclasses import dataclass

# Helper
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

#2D Drone Physics

@dataclass
class Drone2D:
    # Physical params
    m: float = 1.2           # kg
    I: float = 0.03          # kg*m^2 (rotational inertia)
    arm: float = 0.18        # meters (half distance between rotors)
    g: float = 9.81          # m/s^2
    max_thrust: float = 16.0 # N per motor (roughly)

    # State
    x: float = 0.0
    z: float = 1.0
    vx: float = 0.0
    vz: float = 0.0
    theta: float = 0.0       # radians (0 = upright)
    omega: float = 0.0       # rad/s

    # Some damping to keep it nice
    lin_damp: float = 0.02
    ang_damp: float = 0.04

    def step(self, u_left: float, u_right: float, dt: float):
        # Clamp inputs
        u_left = clamp(u_left, 0.0, self.max_thrust)
        u_right = clamp(u_right, 0.0, self.max_thrust)

        # Forces / torques
        T = u_left + u_right
        tau = (u_right - u_left) * self.arm

        # Acceleration (world frame)
      
        ax = -(T / self.m) * math.sin(self.theta)
        az =  (T / self.m) * math.cos(self.theta) - self.g

        # Angular acceleration
        alpha = tau / self.I

        # Semi-implicit Euler integration 
        self.vx += ax * dt
        self.vz += az * dt
        self.omega += alpha * dt

        # Damping
        self.vx *= (1.0 - self.lin_damp)
        self.vz *= (1.0 - self.lin_damp)
        self.omega *= (1.0 - self.ang_damp)

        self.x += self.vx * dt
        self.z += self.vz * dt
        self.theta = wrap_angle(self.theta + self.omega * dt)

        # Ground collision (simple)
        if self.z < 0.0:
            self.z = 0.0
            if self.vz < 0:
                self.vz = 0.0
            # optional: kill spin on impact
            self.omega *= 0.5


# Simple PD "autopilot" to hold a target (x,z)

@dataclass
class Controller:
    # Outer loop gains (position -> desired accel)
    kp_x: float = 1.6
    kd_x: float = 2.2
    kp_z: float = 5.0
    kd_z: float = 3.5

    # Inner loop gains (attitude)
    kp_theta: float = 14.0
    kd_theta: float = 2.8

    max_tilt: float = math.radians(35)   # max pitch angle
    max_az: float = 8.0                  # max vertical accel command (m/s^2)
    max_ax: float = 6.0                  # max horizontal accel command (m/s^2)

    def compute(self, drone: Drone2D, x_ref: float, z_ref: float):
        # Position error
        ex = x_ref - drone.x
        ez = z_ref - drone.z

        # Desired accelerations (PD)
        ax_des = self.kp_x * ex - self.kd_x * drone.vx
        az_des = self.kp_z * ez - self.kd_z * drone.vz

        ax_des = clamp(ax_des, -self.max_ax, self.max_ax)
        az_des = clamp(az_des, -self.max_az, self.max_az)

        # Map desired horizontal accel -> desired tilt angle
        # For small angles: ax ≈ -g * theta  => theta_des ≈ -ax_des/g
        theta_des = clamp(-ax_des / drone.g, -self.max_tilt, self.max_tilt)

        # Total thrust to achieve vertical accel while tilted
        # az = (T/m)*cos(theta) - g  => T = m*(g + az_des)/cos(theta)
        c = max(0.2, math.cos(drone.theta))
        T_des = drone.m * (drone.g + az_des) / c
        T_des = clamp(T_des, 0.0, 2.0 * drone.max_thrust)

        # Attitude PD -> desired torque
        e_theta = wrap_angle(theta_des - drone.theta)
        tau_des = self.kp_theta * e_theta - self.kd_theta * drone.omega

        # Convert (T, tau) -> (u_left, u_right)
        # tau = (uR - uL)*arm and T = uL + uR
        u_right = 0.5 * (T_des + tau_des / drone.arm)
        u_left  = 0.5 * (T_des - tau_des / drone.arm)

        # Clamp motor thrusts
        u_left = clamp(u_left, 0.0, drone.max_thrust)
        u_right = clamp(u_right, 0.0, drone.max_thrust)

        return u_left, u_right, theta_des, T_des


# Visualization (pygame)

def world_to_screen(x, z, w, h, scale, x0, z0):
    # world (x right, z up) -> screen (x right, y down)
    sx = int(w * 0.5 + (x - x0) * scale)
    sy = int(h - 80 - (z - z0) * scale)
    return sx, sy

def draw_drone(screen, drone: Drone2D, w, h, scale, x0, z0):
    # Drone body as a line/rod with two motor points
    cx, cy = world_to_screen(drone.x, drone.z, w, h, scale, x0, z0)

    # rotor positions in world
    dx = drone.arm * math.cos(drone.theta)
    dz = drone.arm * math.sin(drone.theta)

    # left rotor world
    lx = drone.x - dx
    lz = drone.z - dz
    # right rotor world
    rx = drone.x + dx
    rz = drone.z + dz

    slx, sly = world_to_screen(lx, lz, w, h, scale, x0, z0)
    srx, sry = world_to_screen(rx, rz, w, h, scale, x0, z0)

    # Body line
    pygame.draw.line(screen, (230, 230, 230), (slx, sly), (srx, sry), 5)

    # Center
    pygame.draw.circle(screen, (255, 200, 80), (cx, cy), 6)

    # Rotors
    pygame.draw.circle(screen, (80, 200, 255), (slx, sly), 7)
    pygame.draw.circle(screen, (80, 200, 255), (srx, sry), 7)

    # Heading indicator (a small line upwards from center)
    tipx = drone.x - 0.22 * math.sin(drone.theta)
    tipz = drone.z + 0.22 * math.cos(drone.theta)
    stx, sty = world_to_screen(tipx, tipz, w, h, scale, x0, z0)
    pygame.draw.line(screen, (255, 120, 120), (cx, cy), (stx, sty), 3)

def main():
    pygame.init()
    w, h = 1100, 700
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Simple Drone Physics (2D) - pygame")
    font = pygame.font.SysFont("consolas", 18)

    clock = pygame.time.Clock()

    drone = Drone2D()
    ctrl = Controller()

    # Camera reference
    x0, z0 = 0.0, 0.0
    scale = 120.0  # px per meter

    # Target
    x_ref, z_ref = 0.0, 1.5

    autopilot = True
    paused = False

    # Manual fallback 
    uL = drone.m * drone.g / 2.0
    uR = drone.m * drone.g / 2.0

    # Fixed-step simulation for stability
    sim_dt = 1.0 / 120.0
    accumulator = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        accumulator += dt

        # ----- events / input -----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    autopilot = not autopilot
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    drone = Drone2D()
                    x_ref, z_ref = 0.0, 1.5
                    uL = drone.m * drone.g / 2.0
                    uR = drone.m * drone.g / 2.0

        keys = pygame.key.get_pressed()
        # Move target 
        if keys[pygame.K_LEFT]:
            x_ref -= 1.2 * dt
        if keys[pygame.K_RIGHT]:
            x_ref += 1.2 * dt
        if keys[pygame.K_UP]:
            z_ref += 1.2 * dt
        if keys[pygame.K_DOWN]:
            z_ref -= 1.2 * dt
        z_ref = clamp(z_ref, 0.2, 6.0)

        # Manual controls 
        if not autopilot:
            # W/S for collective thrust, Q/E for differential
            if keys[pygame.K_w]:
                uL += 20.0 * dt
                uR += 20.0 * dt
            if keys[pygame.K_s]:
                uL -= 20.0 * dt
                uR -= 20.0 * dt
            if keys[pygame.K_q]:
                uL += 15.0 * dt
                uR -= 15.0 * dt
            if keys[pygame.K_e]:
                uL -= 15.0 * dt
                uR += 15.0 * dt

            uL = clamp(uL, 0.0, drone.max_thrust)
            uR = clamp(uR, 0.0, drone.max_thrust)

        # ----- simulation -----
        if not paused:
            while accumulator >= sim_dt:
                if autopilot:
                    uL, uR, theta_des, T_des = ctrl.compute(drone, x_ref, z_ref)
                drone.step(uL, uR, sim_dt)
                accumulator -= sim_dt
        else:
            theta_des, T_des = 0.0, 0.0

        
        screen.fill((18, 18, 22))

        # Ground
        pygame.draw.line(screen, (70, 70, 70), (0, h - 80), (w, h - 80), 3)

        # Target marker
        tx, ty = world_to_screen(x_ref, z_ref, w, h, scale, x0, z0)
        pygame.draw.circle(screen, (120, 255, 120), (tx, ty), 8, 2)
        pygame.draw.line(screen, (120, 255, 120), (tx - 12, ty), (tx + 12, ty), 2)
        pygame.draw.line(screen, (120, 255, 120), (tx, ty - 12), (tx, ty + 12), 2)

        draw_drone(screen, drone, w, h, scale, x0, z0)

        # HUD text
        lines = [
            f"Autopilot [A]: {autopilot} | Pause [SPACE]: {paused} | Reset [R]",
            "Move target: ARROWS | Manual (autopilot off): W/S thrust, Q/E tilt",
            f"x={drone.x: .2f} m   z={drone.z: .2f} m   vx={drone.vx: .2f}   vz={drone.vz: .2f}",
            f"theta={math.degrees(drone.theta): .1f} deg   omega={math.degrees(drone.omega): .1f} deg/s",
            f"uL={uL: .2f} N   uR={uR: .2f} N   target=({x_ref: .2f}, {z_ref: .2f})",
        ]
        y = 12
        for s in lines:
            surf = font.render(s, True, (220, 220, 220))
            screen.blit(surf, (12, y))
            y += 22

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
