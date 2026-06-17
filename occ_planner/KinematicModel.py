# import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
class Vehicle:
    def __init__(self):
        self.lw = 2.8  # wheelbase
        self.lf = 0.96  # front hang length
        self.lr = 0.929  # rear hang length
        self.lb = 1.942  # width
        self.MAX_STEER = 0.75
        self.MAX_SPEED = 10/3.6
        self.MIN_SPEED = 0
        self.MAX_ACC = 1.5
        self.MAX_STEERING_RATE = np.deg2rad(20)
        self.MAX_T = 0.5
        self.MIN_T = 0.1
        self.Buffer_Radius = np.sqrt(( 0.5 * (self.lr + self.lf + self.lw) )**2 + self.lb**2) / 2 # 缓冲半径
    def create_polygon(self, x, y, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        points = np.array([
            [-self.lr, -self.lb / 2, 1],
            [self.lf + self.lw, -self.lb / 2, 1],
            [self.lf + self.lw, self.lb / 2, 1],
            [-self.lr, self.lb / 2, 1],
            [-self.lr, -self.lb / 2, 1],
        ]).dot(np.array([
            [cos_theta, -sin_theta, x],
            [sin_theta, cos_theta, y],
            [0, 0, 1]
        ]).transpose())
        return np.array(points[:, 0:2])
    
    def draw_polygon(self, x, y, theta):
        """
        绘制车辆多边形
        :param x: 车辆中心点的 x 坐标
        :param y: 车辆中心点的 y 坐标
        :param theta: 车辆朝向角度（弧度）
        :return: None
        """
        polygon = self.create_polygon(x, y, theta)
        plt.plot(polygon[:, 0], polygon[:, 1], 'b-')
    # @staticmethod
    def GetMidState(self,SegLength,ControlCommand,VehState):
        x,y, yaw = VehState
        stepsize,steer = ControlCommand
        DiscreteNum = int(SegLength/stepsize)
        MidState = []
        # 计算中间状态
        for i in range(DiscreteNum):
            x = x + stepsize * np.cos(yaw)
            y = y + stepsize * np.sin(yaw)
            yaw = yaw + stepsize * np.tan(steer) / self.lw
            MidState.append([x,y,yaw])
        return MidState

def calculate_polygon_area(points):
    """
    计算多边形面积的函数
    
    参数:
        points: 包含多边形顶点的列表，每个顶点是一个(x, y)元组
        示例: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    返回:
        多边形的面积 (float)
    """
    n = len(points)  # 顶点数量
    area = 0.0  # 初始化面积
    
    if n < 3:
        raise ValueError("多边形至少需要3个顶点")
    
    # 应用鞋带公式
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]  # 下一个顶点（循环连接）
        area += (x1 * y2 - x2 * y1)
    
    # 返回面积（取绝对值后除以2）
    return abs(area) / 2.0

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    vehicle = Vehicle()
    # Example usage
    x, y, theta = 0.0, 0.0, 0.0
    polygon = vehicle.create_polygon(x, y, theta)
    print("Vehicle Polygon Points:\n", polygon)
    
    # Example mid-state calculation
    seg_length = 5.0
    control_command = (1.0, 0.1)  # stepsize, steer angle
    veh_state = np.array([x, y, theta])
    mid_states = vehicle.GetMidState(seg_length, control_command, veh_state)
    for state in mid_states:
        state = np.array(state)
        temp = vehicle.create_polygon(state[0], state[1], state[2])
        plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'green')
        print("Mid State:", state)
    plt.show()
    print("Mid States:\n", mid_states)