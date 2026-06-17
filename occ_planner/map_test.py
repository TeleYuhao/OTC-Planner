from TPCAP_Cases import Case
import matplotlib.pyplot as plt
import numpy as np
from KinematicModel import Vehicle
from HybridAstar import HybridAstar
from TPCAP_Cases import calculate_corners
from shapely.geometry import Point
from aabbtree import AABB
from aabbtree import AABBTree
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

def MakeGridMap(case:Case, grid_size:float):
    grid_x = int((case.xmax - case.xmin)/ grid_size)
    grid_y = int((case.ymax - case.ymin)/ grid_size)
    GridMap = np.zeros((grid_x, grid_y), dtype=int)
    for x,y in zip(case.discrete_x,case.discrete_y):
        grid_x_index = int((x - case.xmin) / grid_size)
        grid_y_index = int((y - case.ymin) / grid_size)
        if 0 <= grid_x_index < GridMap.shape[0] and 0 <= grid_y_index < GridMap.shape[1]:
            GridMap[grid_x_index, grid_y_index] = 255
            x = grid_x_index * grid_size + case.xmin
            y = grid_y_index * grid_size + case.ymin
            # plt.plot(x,y, 'x', color='k')
    return GridMap
def MakeBufferGridMap(case:Case, grid_size:float):
    grid_x = int((case.xmax - case.xmin)/ grid_size)
    grid_y = int((case.ymax - case.ymin)/ grid_size)
    GridMap = np.zeros((grid_x, grid_y), dtype=int)
    for x in range(grid_x):
        for y in range(grid_y):
            Map_x = case.xmin + x * grid_size + grid_size / 2
            Map_y = case.ymin + y * grid_size + grid_size / 2
            point = Point(Map_x, Map_y)
            if case.BufferedPolygon.contains(point).any():
                GridMap[x, y] = 255
    return GridMap

def ShowMap(case:Case, grid_map:np.ndarray, show=True):
    for x in range(grid_map.shape[0]):
        for y in range(grid_map.shape[1]):
            Map_x = case.xmin + x * case.MapGridSize + case.MapGridSize / 2
            Map_y = case.ymin + y * case.MapGridSize + case.MapGridSize / 2
            if grid_map[x, y] == 255:
                plt.plot(Map_x, Map_y, 'x', color='k')

def test_GridMap():
    i = 10
    for i in range(1,21):
        res = 0.1
        veh = Vehicle()
        map = Case(f'BenchmarkCases/Case{i}.csv',MapgridSize=0.1, discrete_size=0.01)
        grid_map = MakeGridMap(map, grid_size=0.1)
        Cor = HybridAstar(72)
        Cor.Init(grid_map, res, res, map.xmax, map.ymax, map.xmin, map.ymin)

        Expand_Length = Cor.GenerateCorridor([map.xf, map.yf, map.thetaf], 5, res/2, [0,0,0,0])
            # Expand_Length[0] += 0.3
        print(Expand_Length)

        conor = calculate_corners(map.xf, map.yf, map.thetaf, Expand_Length, veh.lf + veh.lw, veh.lr, veh.lb / 2)
        plt.plot(conor[:, 0], conor[:, 1], color='blue',  label='Corners')
        for i in range(len(conor)):
            if i == len(conor) - 1:
                plt.plot([conor[i, 0], conor[0, 0]], [conor[i, 1], conor[0, 1]], color='blue', linewidth=0.5)
            else:
                plt.plot([conor[i, 0], conor[i + 1, 0]], [conor[i, 1], conor[i + 1, 1]], color='blue', linewidth=0.5)
        
        plt.imshow(grid_map.T, 
                        cmap='binary', 
                        origin='lower',
                            aspect='auto',
                            extent=[map.xmin, map.xmax, map.ymin, map.ymax])
        plt.scatter(map.discrete_x,map.discrete_y, color='blue', marker='o', label='Discrete Points')
        map.ShowMap(show=True)

def test_Buffered_Map():
    i = 5
    for i in range(1,21):
        if i == 7: continue
        map = Case(f'BenchmarkCases/Case{i}.csv',MapgridSize=0.1, discrete_size=0.01)
        path = np.load(f'PlanningRes/TPCAP_Case_{i}_Hybrid_A_star.npy')
        res = 0.1

        start = [map.x0, map.y0, map.theta0]
        goal = [map.xf, map.yf, map.thetaf]
        for pose in path:
            LocalCorridor = MakeCorridor(map, pose)
            LocalCorridor = np.vstack((LocalCorridor,LocalCorridor[0]))
            plt.plot(LocalCorridor[:, 0], LocalCorridor[:, 1], color='blue', label='Corners')
        plt.show()

def check_AABB_collision(veh_rec, AABB_ploygon):
    '''
    param: obs_rec、AABB_polygon  以角点形式储存的ndarray
    '''
    min_x = min(veh_rec[:, 0])
    max_x = max(veh_rec[:, 0])
    min_y = min(veh_rec[:, 1])
    max_y = max(veh_rec[:, 1])
    
    veh_AABB = AABB(((min_x, max_x), (min_y, max_y)))
    return AABB_ploygon.overlaps(veh_AABB)
        

def MakeLIOMCorridor(case:Case,point,ds=0.1):
    bound = np.array([1e-4, 1e-4, 1e-4, 1e-4])
    dirs = [0, 1, 2, 3]
    point = point[:2]
    obs_rec = point - np.array([[bound[0], bound[1]],   # 左下
                                [-bound[2], bound[1]],  # 右下
                                [-bound[2], -bound[3]], # 右上
                                [bound[0], -bound[3]]]) # 左上
    COLLISION = [True] * 4
    while any(COLLISION):
        if (bound > 4).any():
            return obs_rec
        for i in range(len(dirs)):
            if COLLISION[i] == False: continue
            bound[i] += ds
            new_vec = np.array([
                                [bound[0] ,  bound[1]],
                                [-bound[2],  bound[1]],
                                [-bound[2], -bound[3]],
                                [bound[0] , -bound[3]]
                                ])
            obs_rec = point - new_vec
            for obs_index in range(len(case.obs)):
                if check_AABB_collision(obs_rec, case.AABB_Buffered_Polygon[obs_index]):
                    if case.BufferedPolygon['geometry'][obs_index].intersects(Polygon(obs_rec)):
                        COLLISION[i] = False
                        bound[i] -= ds
                        break
            
    return obs_rec
            


if __name__ == "__main__":
    test_Buffered_Map()
    