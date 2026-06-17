import numpy as np
import matplotlib.pyplot as plt
from Optimize_util import downsample_trajectory


def analyze_vehicle_direction(trajectory, tolerance_deg=15):
    """修正版：正确区分前进/后退，并确保分段端点重合"""
    if len(trajectory) < 2:
        return [(0, 0, 1)] if len(trajectory) == 1 else []
    
    traj = np.array(trajectory)
    x, y, yaw = traj[:, 0], traj[:, 1], traj[:, 2]
    
    # 计算位移向量
    dx = np.diff(x)
    dy = np.diff(y)
    displacement_angles = np.arctan2(dy, dx)
    
    # 计算位移方向与车辆朝向的最小角度差
    angle_diff = np.abs((displacement_angles - yaw[:-1] + np.pi) % (2 * np.pi) - np.pi)
    
    # 判断前进或后退
    tolerance_rad = np.radians(tolerance_deg)
    directions = np.zeros(len(angle_diff))
    directions[angle_diff <= tolerance_rad] = 1   # 前进
    directions[angle_diff >= np.pi - tolerance_rad] = -1 # 后退
    
    # 处理不确定段：继承前一段方向
    for i in range(len(directions)):
        if directions[i] == 0:
            if i == 0:
                directions[i] = 1
            else:
                directions[i] = directions[i-1]
    
    # ================= 核心修复部分 =================
    # 切分连续段，确保端点重合
    segments = []
    current_dir = directions[0]
    start_idx = 0
    
    # directions 的长度是 N-1，索引 i 代表点 i 到 i+1 的位移
    for i in range(1, len(directions)):
        if directions[i] != current_dir:
            # 修复：前一段的终点应该是 i（而不是 i-1），因为 directions[i-1] 是点 i-1 到 i 的位移
            segments.append((start_idx, i, current_dir)) 
            # 下一段的起点也是 i，实现端点重合
            start_idx = i 
            current_dir = directions[i]
            
    # 添加最后一段，终点是最后一个轨迹点
    segments.append((start_idx, len(trajectory) - 1, current_dir))
    # ================================================
    
    return segments

def compute_arc_lengths(trajectory):
    """计算轨迹点间的累积弧长（距离）"""
    traj = np.array(trajectory)
    x, y = traj[:, 0], traj[:, 1]
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.zeros(len(trajectory))
    s[1:] = np.cumsum(distances)
    return s

def generate_trapezoidal_profile(distance, max_velocity, acceleration, num_points):
    """
    为单段距离生成梯形速度曲线
    
    参数:
        distance: 段的总长度
        max_velocity: 最大速度
        acceleration: 加速度/减速度
        num_points: 需要生成的数据点数量
    
    返回:
        times: 时间数组 (num_points,)
        velocities: 速度数组 (num_points,)
        accelerations: 加速度数组 (num_points,)
    """
    if distance <= 0 or num_points <= 0:
        return np.array([]), np.array([]), np.array([])
    
    # 1. 计算加速到最大速度所需的时间和距离
    t_acc = max_velocity / acceleration
    s_acc = 0.5 * acceleration * t_acc**2
    
    # 2. 判断是梯形(有匀速段)还是三角形(无匀速段)
    if 2 * s_acc >= distance:
        # 三角形模式：无法达到最大速度，加速后立即减速
        # 计算能达到的最大速度 v_max_real
        v_max_real = np.sqrt(acceleration * distance / 2)
        t_acc_real = v_max_real / acceleration
        t_total = 2 * t_acc_real
        
        # 生成时间点
        times = np.linspace(0, t_total, num_points)
        
        # 计算速度和加速度
        velocities = np.zeros(num_points)
        accelerations = np.zeros(num_points)
        
        for i, t in enumerate(times):
            if t < t_acc_real:
                # 加速阶段
                velocities[i] = acceleration * t
                accelerations[i] = acceleration
            else:
                # 减速阶段
                velocities[i] = v_max_real - acceleration * (t - t_acc_real)
                accelerations[i] = -acceleration
                
    else:
        # 梯形模式：有匀速段
        s_const = distance - 2 * s_acc # 匀速段距离
        t_const = s_const / max_velocity
        t_total = 2 * t_acc + t_const
        
        # 生成时间点
        times = np.linspace(0, t_total, num_points)
        
        # 计算速度和加速度
        velocities = np.zeros(num_points)
        accelerations = np.zeros(num_points)
        
        for i, t in enumerate(times):
            if t < t_acc:
                # 加速阶段
                velocities[i] = acceleration * t
                accelerations[i] = acceleration
            elif t < t_acc + t_const:
                # 匀速阶段
                velocities[i] = max_velocity
                accelerations[i] = 0
            else:
                # 减速阶段
                t_dec = t - (t_acc + t_const)
                velocities[i] = max_velocity - acceleration * t_dec
                accelerations[i] = -acceleration
    
    return times, velocities, accelerations

def plan_velocity_profiles(trajectory, segments, max_velocity=10/3.6, acceleration=1.5):
    """
    为整个轨迹生成速度规划
    
    返回:
        all_times: 完整的时间序列 (与轨迹点一一对应)
        all_velocities: 完整的速度序列
        all_accelerations: 完整的加速度序列
    """
    arc_lengths = compute_arc_lengths(trajectory)
    total_points = len(trajectory)
    
    all_times = np.zeros(total_points)
    all_velocities = np.zeros(total_points)
    all_accelerations = np.zeros(total_points)
    
    current_global_time = 0.0
    
    for i, (start_idx, end_idx, direction) in enumerate(segments):
        # 1. 获取当前段的长度和点数
        seg_length = arc_lengths[end_idx] - arc_lengths[start_idx]
        num_points_in_segment = end_idx - start_idx + 1
        
        # 2. 为当前段生成速度曲线 (相对时间)
        # 注意：这里假设每一段都是从静止加速到静止减速
        seg_times, seg_velocities, seg_accelerations = generate_trapezoidal_profile(
            distance=seg_length,
            max_velocity=max_velocity,
            acceleration=acceleration,
            num_points=num_points_in_segment
        )
        
        # 3. 应用方向 (后退时速度为负)
        seg_velocities = seg_velocities * direction
        seg_accelerations = seg_accelerations * direction
        
        # 4. 计算时间偏移 (累加到上一段的结束时间)
        # seg_times 是从0开始的，需要加上 current_global_time
        seg_times = seg_times + current_global_time
        
        # 5. 将当前段的数据填入全局数组
        # 确保索引正确
        indices = slice(start_idx, end_idx + 1)
        all_times[indices] = seg_times
        all_velocities[indices] = seg_velocities
        all_accelerations[indices] = seg_accelerations
        
        # 6. 更新全局时间 (用于下一段的偏移)
        # 如果当前段有数据，则更新为该段最后一个时间点
        if len(seg_times) > 0:
            current_global_time = seg_times[-1]
    
    return all_times, all_velocities, all_accelerations

def visualize_velocity_profiles(trajectory, segments, times, velocities, accelerations):
    """可视化速度/加速度剖面"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. 轨迹可视化
    traj = np.array(trajectory)
    x, y = traj[:, 0], traj[:, 1]
    ax1.plot(x, y, 'gray', linestyle='--', alpha=0.3)
    
    colors = {1: 'blue', -1: 'red'}
    for start, end, direction in segments:
        ax1.plot(x[start:end+1], y[start:end+1], color=colors[direction], linewidth=2.5)
        ax1.scatter(x[start], y[start], s=80, color=colors[direction], zorder=5)
        ax1.scatter(x[end], y[end], s=80, color=colors[direction], marker='s', zorder=5)
    
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.set_title('轨迹与运动方向')
    ax1.grid(True, alpha=0.7)
    ax1.axis('equal')
    
    # 2. 速度/加速度剖面
    ax2.plot(times, velocities, 'b-', linewidth=2, label='速度 (m/s)')
    ax2.plot(times, accelerations, 'r--', linewidth=1.5, label='加速度 (m/s²)')
    
    # 标记段边界
    for start, end, direction in segments:
        if start < len(times):
            ax2.axvline(x=times[start], color='k', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('值')
    ax2.set_title('梯形速度规划剖面')
    ax2.grid(True, alpha=0.7)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 假设这是您的车辆参数
class Config:
    lw = 2.8  # 轴距 (Wheelbase)，请替换为您实际的 C.lw 值
C = Config()

def compute_curvature_and_steering(trajectory, times):
    """
    根据离散轨迹点计算曲率、前轮转角和前轮转角变化率
    
    参数:
        trajectory: 形状为 (N, 3) 的数组，包含 [x, y, theta]
        times: 形状为 (N,) 的时间数组
    
    返回:
        curvatures: 曲率数组 (N,)
        steerings: 前轮转角数组 (N,)
        steering_rates: 前轮转角变化率数组 (N,)
    """
    traj = np.array(trajectory)
    x, y = traj[:, 0], traj[:, 1]
    
    # 1. 计算弧长 s
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.sqrt(dx**2 + dy**2)
    # 防止除以0（如果存在完全重合的点）
    ds[ds < 1e-6] = 1e-6 
    
    # 2. 计算一阶导数 (dx/ds, dy/ds)
    x_prime = dx / ds
    y_prime = dy / ds
    
    # 3. 计算二阶导数 (d^2x/ds^2, d^2y/ds^2)
    # 注意：这里对 s 的累积求导更准确
    s = np.zeros(len(x))
    s[1:] = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    
    x_double_prime = np.gradient(x_prime, s)
    y_double_prime = np.gradient(y_prime, s)
    
    # 4. 计算曲率 kappa = (x'*y'' - y'*x'') / (x'^2 + y'^2)^(3/2)
    # 因为 x'^2 + y'^2 = 1，所以分母为 1
    curvatures = x_prime * y_double_prime - y_prime * x_double_prime
    
    # 5. 计算前轮转角 delta = arctan(kappa * L)
    steerings = np.arctan(curvatures * C.lw)
    
    # 6. 计算前轮转角变化率 delta_dot = d(delta)/dt
    # 使用 np.gradient 对时间 t 求导
    times = np.array(times)
    dt = np.gradient(times)
    dt[dt < 1e-6] = 1e-6  # 防止时间差为0
    steering_rates = np.gradient(steerings, times)
    
    return curvatures, steerings, steering_rates

def down_sample(trajectory, target_num=100):
    trajectory = downsample_trajectory(trajectory, target_num=target_num)
    
    # 1. 分析方向
    segments = analyze_vehicle_direction(trajectory, tolerance_deg=10)
    
    # 2. 规划速度剖面
    all_times, all_velocities, all_accelerations = plan_velocity_profiles(
        trajectory, segments, max_velocity=10/3.6, acceleration=1.5
    )
    curvatures, steerings, steering_rates = compute_curvature_and_steering(
        trajectory, all_times
    )
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    yaw = trajectory[:, 2]
    init_res = np.vstack((x,y,all_velocities,yaw,steerings)).T
    init_control = np.vstack((all_accelerations,steering_rates)).T
    Tf = all_times[-1]
    return init_res,init_control[:-1], Tf

def main():
    
    # 模拟加载数据
    i = 12
    trajectory = np.load(f'PlanningRes/TPCAP_Case_{i}_Hybrid_A_star.npy')
    trajectory = downsample_trajectory(trajectory, target_num=100)
    
    # 1. 分析方向
    segments = analyze_vehicle_direction(trajectory, tolerance_deg=10)
    init_res, init_control, Tf = down_sample(trajectory, target_num=100)
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    axs[0].plot(init_res[:,0], init_res[:,1], 'o-', label='降采样轨迹')
    axs[0].set_xlabel('X 坐标')
    axs[0].set_ylabel('Y 坐标')
    axs[0].set_title('降采样轨迹与运动方向')
    axs[0].grid(True, alpha=0.7)
    axs[0].axis('equal')

    axs[1].plot(init_res[:,2], 'b-', linewidth=2, label='速度 v (m/s)')
    axs[1].set_ylabel('速度 (m/s)')
    axs[1].grid(True, alpha=0.5)
    axs[1].legend()

    axs[2].plot(init_res[:,3], 'g-', linewidth=2, label='前轮转角 δ (deg)')
    axs[2].set_ylabel('转角 (度)')
    axs[2].grid(True, alpha=0.5)
    axs[2].legend()

    axs[3].plot(init_control[:,0], 'r-', linewidth=2, label='加速度 a (m/s²)')
    axs[3].set_ylabel('加速度 (m/s²)')
    axs[3].grid(True, alpha=0.5)
    axs[3].legend()
    axs[3].set_xlabel('时间 (s)')


    # 2. 规划速度剖面
    all_times, all_velocities, all_accelerations = plan_velocity_profiles(
        trajectory, segments, max_velocity=10/3.6, acceleration=1.5
    )
    curvatures, steerings, steering_rates = compute_curvature_and_steering(
        trajectory, all_times
    )
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    # 图1: 速度 v
    axs[0].plot(all_times, all_velocities, 'b-', linewidth=2, label='速度 v (m/s)')
    axs[0].set_ylabel('速度 (m/s)')
    axs[0].grid(True, alpha=0.5)
    axs[0].legend()
    
    # 图2: 加速度 a (控制量 u1)
    axs[1].plot(all_times, all_accelerations, 'r-', linewidth=2, label='加速度 a (m/s²)')
    axs[1].set_ylabel('加速度 (m/s²)')
    axs[1].grid(True, alpha=0.5)
    axs[1].legend()
    
    # 图3: 前轮转角 delta (状态量)
    axs[2].plot(all_times, np.degrees(steerings), 'g-', linewidth=2, label='前轮转角 δ (deg)')
    axs[2].set_ylabel('转角 (度)')
    axs[2].grid(True, alpha=0.5)
    axs[2].legend()
    
    # 图4: 前轮转角变化率 delta_dot (控制量 u2)
    axs[3].plot(all_times, steering_rates, 'm-', linewidth=2, label='转角变化率 δ_dot (rad/s)')
    axs[3].set_ylabel('变化率 (rad/s)')
    axs[3].set_xlabel('时间 (s)')
    axs[3].grid(True, alpha=0.5)
    axs[3].legend()
    
    plt.suptitle('基于自行车运动学模型的完整状态与控制量剖面', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 3. 验证输出
    print(f"轨迹点数: {len(trajectory)}")
    print(f"时间戳数量: {len(all_times)}")
    print(f"速度点数量: {len(all_velocities)}")
    print(f"加速度点数量: {len(all_accelerations)}")
    
    # 4. 可视化
    visualize_velocity_profiles(trajectory, segments, all_times, all_velocities, all_accelerations)

if __name__ == "__main__":
    main()