from KinematicModel import Vehicle
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from shapely.geometry import Polygon
import geopandas as gpd
from aabbtree import AABB
import polytope as pt
import matplotlib.gridspec as gridspec
plt.rcParams['axes.unicode_minus'] = False
class Case:
    def __init__(self,Filename:str,
                 discrete_size:float,
                 MapgridSize:float) -> None:
        self.vehicle = Vehicle()
        self.discrete_size = discrete_size
        self.MapGridSize = MapgridSize
        with open(Filename, 'r') as f:
            reader = csv.reader(f)
            tmp = list(reader)
            v = [float(i) for i in tmp[0]]
            self.x0, self.y0, self.theta0 = v[0:3]
            self.xf, self.yf, self.thetaf = v[3:6]
            self.obs_num = int(v[6])
            num_vertexes = np.array(v[7:7 + self.obs_num], dtype=np.int16)
            vertex_start = 7 + self.obs_num + (np.cumsum(num_vertexes, dtype=np.int16) - num_vertexes) * 2
            self.obs = []
            for vs, nv in zip(vertex_start, num_vertexes):
                self.obs.append(np.array(v[vs:vs + nv * 2]).reshape((nv, 2), order='A'))
                # self.obs.append(np.array(v[vs:vs + nv * 2],dtype=np.double).reshape((nv, 2)))
        self.CheckSize()
        self.DiscreteMap()
        self.MakeBufferPolygon()
        self.GetConvexObs()
    def GetStart(self):
        return np.array([self.x0, self.y0, self.theta0])
    def GetGoal(self):
        return np.array([self.xf, self.yf, self.thetaf])
    def GetConvexObs(self):
        self.ConvexObs = []
        for obs in self.obs:
            hull = pt.qhull(obs).vertices
            center = np.mean(hull, axis=0)
            angles = np.arctan2(hull[:,1] - center[1], hull[:,0] - center[0])
            self.ConvexObs.append(hull[np.argsort(angles)])
    def CheckSize(self):
        # if self.xmin>1e3 and self.xmax >1e3 and abs(self.ymin) > 1e3 and abs(self.ymin) > 1e3:
        base_x,base_y = np.inf,np.inf
        max_x , max_y = -np.inf,-np.inf
        for i in range(self.obs_num):
            base_x = min(self.obs[i][:, 0].min(),base_x,self.x0,self.xf)
            base_y = min(self.obs[i][:, 1].min(),base_y,self.y0,self.yf)
            max_x = max(self.obs[i][:, 0].max(),max_x, self.x0, self.xf)
            max_y = max(self.obs[i][:, 1].max(),max_y, self.y0, self.yf)
        gap = 6
        self.x0 = self.x0 - base_x + gap
        self.y0 = self.y0 - base_y + gap
        self.xf = self.xf - base_x + gap
        self.yf = self.yf - base_y + gap
        for i in range(self.obs_num):
            self.obs[i][:, 0] =  self.obs[i][:, 0] - base_x + gap
            self.obs[i][:, 1] =  self.obs[i][:, 1] - base_y + gap
        self.base_x = base_x - gap
        self.base_y = base_y - gap
        self.xmin ,self.ymin = 0, 0
        self.xmax ,self.ymax = math.ceil(max_x - base_x  + 16), math.ceil(max_y - base_y  + 16)
    def MakeBufferPolygon(self):
        geometries = [Polygon(obs) for obs in self.obs]
        # 正确创建GeoDataFrame并设置几何列
        gdf = gpd.GeoDataFrame({'geometry': geometries})
        buffered_geometries = gdf.geometry.buffer(self.vehicle.Buffer_Radius)
        self.BufferedPolygon = gpd.GeoDataFrame(geometry=buffered_geometries)
        self.get_AABB_Buffered_polygon()
        # self.BufferedPolygon.plot()
    def get_AABB_Buffered_polygon(self):
        '''
        func: 获得polygon的AABB
        param：polygon list：polygon的列表
        '''
        self.AABB_Buffered_Polygon = []
        for obs_polygon in self.BufferedPolygon['geometry']:
            max_x = max(obs_polygon.exterior.xy[0])
            min_x = min(obs_polygon.exterior.xy[0])
            max_y = max(obs_polygon.exterior.xy[1])
            min_y = min(obs_polygon.exterior.xy[1])
            self.AABB_Buffered_Polygon.append(AABB([(min_x, max_x), (min_y, max_y)]))
    def DiscreteMap(self):
        self.discrete_x = []
        self.discrete_y = []
        for i in range(self.obs_num):
            obs = self.obs[i]
            for j in range(len(obs)):
                if j == len(obs) - 1:
                    point_1 = obs[j]
                    point_2 = obs[0]
                else :
                    point_1 = obs[j]
                    point_2 = obs[j + 1]
                distance = np.linalg.norm(point_2 - point_1)
                num_points = int(np.ceil(distance / self.discrete_size))
                dx = np.linspace(point_1[0], point_2[0], num_points)
                dy = np.linspace(point_1[1], point_2[1], num_points)
                if len(self.discrete_x) == 0:
                    self.discrete_x = dx
                    self.discrete_y = dy
                else:
                    self.discrete_x = np.concatenate((self.discrete_x, dx))
                    self.discrete_y = np.concatenate((self.discrete_y, dy))
        self.GridMap = make_octomap(self.discrete_x, self.discrete_y,self.MapGridSize)
    def plotLine(self,GridIndex):
        for Grid in GridIndex:
            plt.plot(Grid[0] * self.discrete_size , Grid[1] * self.discrete_size, 'x', color='k')
    def ShowMap(self,i=0,show=False):
        padding = 0
        plt.xlabel('X / m', fontsize = 27)
        plt.ylabel('Y / m', fontsize = 27)
        for j in range(0, self.obs_num):
            plt.fill(self.obs[j][:, 0], self.obs[j][:, 1], facecolor = 'k', alpha = 0.5)

        plt.arrow(self.x0, self.y0, np.cos(self.theta0), np.sin(self.theta0), width=0.2, color = "gold")
        plt.arrow(self.xf, self.yf, np.cos(self.thetaf), np.sin(self.thetaf), width=0.2, color = "gold")
        temp = self.vehicle.create_polygon(self.x0, self.y0, self.theta0)
        # plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'green',label='Start')
        # temp = self.vehicle.create_polygon(self.xf, self.yf, self.thetaf)
        # plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'red',label="Goal")
        plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'green')
        temp = self.vehicle.create_polygon(self.xf, self.yf, self.thetaf)
        plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'red')
        # plt.legend()
        if show:
            plt.show()
        # plt.show()
    def ShowRes(self,RawPath,Path,Control,i,save=False,show=False,Benchmark="LIOM"):
        veh = Vehicle()
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 2, 
                       width_ratios=[2, 1],  # 主图宽度是右侧子图的2倍
                       height_ratios=[1, 1], # 上下子图高度相等
                       wspace=0.2,           # 列间距
                       hspace=0
                    )
        ax1 = plt.subplot(gs[:, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 1])
        ax1.set_xlim(min(Path[:,0]) - 5, max(Path[:,0]) + 5)
        ax1.set_ylim(min(Path[:,1]) - 5, max(Path[:,1]) + 5)
        # ax1.set_aspect('equal', adjustable = 'box')
        # ax1.gca().set_axisbelow(True)
        # ax1.set_title('Case %d' % (i + 1),fontsize = 27)
        # ax1.grid(linewidth = 0.2)
        ax1.set_xlabel('X / m', fontsize = 27)
        ax1.set_ylabel('Y / m', fontsize = 27)
        
        for j in range(0, self.obs_num):
            ax1.fill(self.obs[j][:, 0], self.obs[j][:, 1], facecolor = 'k', alpha = 0.5)
        for pose in Path:
            temp = veh.create_polygon(pose[0], pose[1], pose[2])
            ax1.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.4, alpha=0.5, color='blue')
        ax1.arrow(self.x0, self.y0, np.cos(self.theta0), np.sin(self.theta0), width=0.2, color = "gold")
        ax1.arrow(self.xf, self.yf, np.cos(self.thetaf), np.sin(self.thetaf), width=0.2, color = "gold")
        temp = self.vehicle.create_polygon(self.x0, self.y0, self.theta0)
        ax1.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'green')
        temp = self.vehicle.create_polygon(self.xf, self.yf, self.thetaf)
        ax1.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4,color = 'red')
        ax1.plot(Path[:,0],Path[:,1],linewidth=2,color="green",label="Smoothed")
        ax1.plot(RawPath[:,0],RawPath[:,1],linewidth=2,color="red",label="Hybrid A*")
        ax1.legend(fontsize=20)
        # ax2.plot(np.cumsum(Control[:,2][:-1]),Control[:,0])
        # time_line =  np.insert(np.cumsum(Control[:,2]), 0, 0)
        if len(Control) != 100:
            time_line = np.linspace(0,Control[-1],len(Path)).flatten()
        else:
            time_line = Control[:,2]
        ax2.plot(time_line,Path[:,3])
        ax2.plot(time_line, np.ones_like(time_line)*self.vehicle.MAX_SPEED,linestyle='--', linewidth = 0.4, color = 'red')
        ax2.plot(time_line,-np.ones_like(time_line)*self.vehicle.MAX_SPEED,linestyle='--', linewidth = 0.4, color = 'red')
        ax2.set_xlabel('t/s', fontsize = 27)
        ax2.set_ylabel('v/ $m/s$', fontsize = 27)
        ax3.set_xlabel('t / s', fontsize = 27)
        ax3.set_ylabel('$\phi$ / $rad$', fontsize = 27)
        ax3.plot(time_line,np.ones_like(time_line)*self.vehicle.MAX_STEER,linestyle='--', linewidth = 0.4, color = 'red')
        ax3.plot(time_line,-np.ones_like(time_line)*self.vehicle.MAX_STEER,linestyle='--', linewidth = 0.4, color = 'red')
        # ax3.plot(np.cumsum(Control[:,2]),Control[:,1])
        ax3.plot(time_line,Path[:,-1])
        plt.subplots_adjust(left=0.08, right=0.96, 
                   bottom=0.1, top=0.92,
                   hspace=0.05)  # 精确控制边界
        plt.tight_layout()
        if save:
            plt.savefig(f'Figure/{Benchmark}_Figure/Case{i}.svg',  bbox_inches='tight', pad_inches=0.1,dpi=300)
        if show:
            plt.show()

def make_octomap(ox, oy, res=0.5):
    df = pd.DataFrame({'x': ox , 'y': oy })  # 单位转为米
    df = df[(df['x'] >= 0) & (df['y'] >= 0)]

    df['x_grid'] = ((df['x']) / res).round().astype(int)
    df['y_grid'] = ((df['y']) / res).round().astype(int)

    grid_binary = df.groupby(['x_grid', 'y_grid']).size().gt(0).astype(int).unstack(fill_value=0)
    grid_binary = grid_binary.T  # 显示时行表示 y，列表示 x

    return np.array(grid_binary)
import numpy as np
import math

def calculate_corners(x, y, yaw, ExpandLength, axleToFront = 2.8+0.96, axleToBack = 0.929, half_width = 0.971):
    """
    计算车辆拓展后的四个角点坐标
    
    参数:
        x, y: 车辆中心点坐标
        yaw: 车辆朝向（弧度）
        ExpandLength: 拓展长度数组 [前, 右, 后, 左]
        axleToFront: 车辆中心到前端的距离
        axleToBack: 车辆中心到后端的距离
        half_width: 车辆半宽
    
    返回:
        numpy数组 (4, 2)，包含四个角点的坐标
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 计算四个角点的局部坐标（相对于车辆中心）
    # 前左角
    front_left = [axleToFront + ExpandLength[0], half_width + ExpandLength[3]]
    # 前右角
    front_right = [axleToFront + ExpandLength[0], -half_width - ExpandLength[1]]
    # 后右角
    rear_right = [-axleToBack - ExpandLength[2], -half_width - ExpandLength[1]]
    # 后左角
    rear_left = [-axleToBack - ExpandLength[2], half_width + ExpandLength[3]]
    
    # 组合成4x2矩阵
    local_pts = np.array([front_left, front_right, rear_right, rear_left])
    
    # 转换为世界坐标系
    world_pts = []
    for pt in local_pts:
        x_local, y_local = pt
        # 旋转变换
        x_rot = x_local * cos_yaw - y_local * sin_yaw
        y_rot = x_local * sin_yaw + y_local * cos_yaw
        # 平移变换
        world_pts.append([x + x_rot, y + y_rot])
    world_pts.append(world_pts[0])  # 闭合多边形
    return np.array(world_pts)

if __name__ == "__main__":
    from HybridAstar import HybridAstar
    veh = Vehicle()
    for i in range(25,100):
        # map = Case(f'BenchmarkCases/Case{i}.csv',MapgridSize=0.1, discrete_size=0.01)
        map = Case(f'LIOM_Benchmark/Case{i}.csv',MapgridSize=0.1, discrete_size=0.01)
        # res = 0.1
        # # grid_binary = make_octomap(map.discrete_x, map.discrete_y, res=0.1)
        # grid_binary = map.GridMap
        # Cor = HybridAstar(72)
        # Cor.Init(grid_binary.T, res, res, map.xmax, map.ymax, -res / 2, -res / 2)
        # Expand_Length = Cor.GenerateCorridor([map.xf, map.yf, map.thetaf], 5, res/2, [0,0,0,0])
        # # Expand_Length[0] += 0.3
        # print(Expand_Length)

        # conor = calculate_corners(map.xf, map.yf, map.thetaf, Expand_Length, veh.lf + veh.lw, veh.lr, veh.lb / 2)
        # plt.plot(conor[:, 0], conor[:, 1], color='blue',  label='Corners')
        # for i in range(len(conor)):
        #     if i == len(conor) - 1:
        #         plt.plot([conor[i, 0], conor[0, 0]], [conor[i, 1], conor[0, 1]], color='blue', linewidth=0.5)
        #     else:
        #         plt.plot([conor[i, 0], conor[i + 1, 0]], [conor[i, 1], conor[i + 1, 1]], color='blue', linewidth=0.5)


        # plt.imshow(grid_binary, 
        #         cmap='binary', 
        #         origin='lower',
        #             aspect='auto',
        #             extent=[map.xmin, map.xmax, map.ymin, map.ymax])
        # plt.scatter(map.discrete_x, map.discrete_y, color='blue', marker='o', label='Discrete Points')
        map.ShowMap(show=True)
    