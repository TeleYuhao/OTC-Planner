import numpy as np

Matrix_R = [[0, 0,0], [0, 0,0],[0,0,1e5]]
Matrix_Q = [[0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    ]
class C:
    lw = 2.8  # wheelbase
    lf = 0.96  # front hang length
    lr = 0.929  # rear hang length
    lb = 1.942  # width
    MAX_STEER = 0.75
    MAX_SPEED = 10/3.6
    MIN_SPEED = 0
    MAX_ACC =  1.5
    MAX_STEERING_RATE = 0.5
    MAX_T = 1
    MIN_T = 0.01
    Length = lf + lw + lr  # total length of the vehicle
    # Buffer_R = 0.4*np.sqrt(((RB+RF)/2)**2+W**2)


def CalVehicleCorner(x, y, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        points = np.array([
            [-C.lr, -C.lb / 2, 1],
            [C.lf + C.lw, -C.lb / 2, 1],
            [C.lf + C.lw, C.lb / 2, 1],
            [-C.lr, C.lb / 2, 1],
        ]).dot(np.array([
            [cos_theta, -sin_theta, x],
            [sin_theta, cos_theta, y],
            [0, 0, 1]
        ]).transpose())
        return np.array(points[:, 0:2])
def Checkyaw(yaw):
    YawChecked = [yaw[0]]
    for i in range(1,len(yaw)):
        if abs(yaw[i] - YawChecked[-1]) > np.pi:
            if yaw[i] > YawChecked[-1]:
                YawChecked.append(yaw[i] - 2 * np.pi)
            else:
                YawChecked.append(yaw[i] + 2 * np.pi)
        else:
            YawChecked.append(yaw[i])
    return YawChecked

def downsample_trajectory(points, target_num=200):
    """
    泊车轨迹降采样（包含位置和航向角）
    
    参数：
        points : 原始轨迹点集合 [[x0, y0, yaw0], [x1, y1, yaw1], ...]
                其中 yaw 为航向角（弧度）
        target_num : 目标采样点数（默认200）
    
    返回：
        降采样后的轨迹点集，格式同输入
    """
    n = len(points)
    if n <= target_num:
        return points
    
    # 1. 分离位置和航向角数据
    positions = np.array([p[:2] for p in points])  # 只取x,y
    yaws = np.array([p[2] for p in points])  # 航向角
    
    # 2. 计算累积路径长度
    diffs = np.diff(positions, axis=0)
    seg_lens = np.hypot(diffs[:,0], diffs[:,1])
    cum_len = np.cumsum(seg_lens)
    total_len = cum_len[-1] if len(cum_len) > 0 else 0
    
    # 3. 处理空轨迹的情况
    if total_len == 0:
        # 等间隔采样航向角
        sampled_indices = np.linspace(0, n-1, target_num, dtype=int)
        return [points[i] for i in sampled_indices]
    
    # 4. 等距采样点（含起点终点）
    sample_distances = np.linspace(0, total_len, target_num)
    
    # 5. 对位置和航向角进行分段插值
    sampled_points = []
    prev_index = 0
    
    for dist in sample_distances:
        # 找到当前距离对应的分段
        idx = prev_index
        while idx < len(cum_len) and cum_len[idx] < dist:
            idx += 1
            
        if idx == 0:
            t = 0.0
        else:
            prev_len = cum_len[idx-1] if idx > 0 else 0
            seg_len = max(cum_len[idx] - prev_len, 1e-5)
            t = (dist - prev_len) / seg_len
            
        # 位置插值
        if idx < len(positions) - 1:
            p1 = positions[idx]
            p2 = positions[idx+1]
            x = p1[0] + t*(p2[0] - p1[0])
            y = p1[1] + t*(p2[1] - p1[1])
        else:
            x, y = positions[-1]
            
        # 航向角插值（使用Slerp球面线性插值）
        if idx < len(yaws) - 1:
            yaw1 = yaws[idx]
            yaw2 = yaws[idx+1]
            
            # 处理角度环绕问题（如从359°到1°）
            angle_diff = yaw2 - yaw1
            if angle_diff > np.pi:
                angle_diff -= 2*np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2*np.pi
                
            yaw = yaw1 + t * angle_diff
            yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # 归一化到[-π, π]
        else:
            yaw = yaws[-1]
        
        sampled_points.append([x, y, yaw])
        prev_index = idx
    
    return np.array(sampled_points)