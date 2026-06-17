from HybridAstar import HybridAstar
from Node import StateNode
from TPCAP_Cases import Case
from config.read_config import read_config
from KinematicModel import Vehicle,calculate_polygon_area
import numpy as np
from TPCAP_Cases import calculate_corners
import matplotlib.pyplot as plt
import heapq
from shapely.geometry import Polygon

class Corridor:
    def GenerateCorridor(self,point,MaxExpandLength=5,delta_s = 0.05,ExpandLength = [0,0,0,0]):
        '''
        Func: Generate Corridor For Vehicle
        '''
        return self.ParkMap.GenerateCorridor(point,MaxExpandLength,delta_s,ExpandLength)
    def calculate_edge_points(self,x,y,yaw,edge_type,ds,ExpandLength):
        return self.ParkMap.calculate_edge_points(x,y,yaw,edge_type,ds,ExpandLength)
    def CalcCorner(self,x,y,yaw):
        '''
        Func: Calculate the four corner points of the vehicle
        '''
        return self.ParkMap.CalcCorner(x,y,yaw)
    def KinematicModel(self,x,y,yaw,delta,StepSize):
        '''
        Func: The Kinematic Model for vehicle
        '''
        return self.ParkMap.DynamicModel(StepSize,delta,x,y,yaw)
    def CollisionCheck(self,state):
        '''
        Func: Check Collision For vehicle
        '''
        return self.ParkMap.CollisionCheck(state[0],state[1],state[2])
    def State2Index(self,state):
        '''
        Func: calc The State Index According Vehicle State
        '''
        return self.ParkMap.State2Index((state[0],state[1],state[2]))
    def GetMidState(self,SegmentLength,StepSize,delta,x,y,yaw):
        return self.ParkMap.GetMidState(SegmentLength,StepSize,delta,x,y,yaw)


class Hybrid_A_Star(Corridor):
    def __init__(self,
                 CorridorMaker:HybridAstar,
                 config:dict,
                 ) -> None:
        super().__init__()
        self.ParkMap = CorridorMaker
        self.config = config
        self.veh = Vehicle()
        self.DiscreteSteeringAngle = config['MaxSteering']/config['SteeringDiscreteNum']
        self.DiscreteNum = config['SteeringDiscreteNum']
        self.ShotDistance  = config['ShotDistance']
        self.SegmentLength = config['SegmentLength']
        self.StepSize = config['StepSize']

        self.open_list = []
        self.STATE_GRID_SIZE_X,self.STATE_GRID_SIZE_Y,self.STATE_GRID_SIZE_PHI = self.ParkMap.GetReso()
        self.closed_list = None
        self.closed_list = [[[None  for _ in range(self.STATE_GRID_SIZE_PHI)]
                                    for _ in range(self.STATE_GRID_SIZE_Y)]
                                    for _ in range(self.STATE_GRID_SIZE_X)]
        self.ForwardMode = True
        self.EmbodyMode  = True
    def Reset(self):
        '''
        func: reset the open_list and close_list
        '''
        self.open_list = []
        self.closed_list = None
        self.closed_list = [[[None  for _ in range(self.STATE_GRID_SIZE_PHI)]
                                    for _ in range(self.STATE_GRID_SIZE_Y)]
                                    for _ in range(self.STATE_GRID_SIZE_X)]
    def Compute_H(self,CurrentNode:StateNode):
        '''
        Function: Calculate the Heuristic cost
        '''
        h = np.linalg.norm(CurrentNode.state[:2] - self.Goal[:2])
        if h < self.ShotDistance:
            h = self.ParkMap.RSLength(CurrentNode.state,self.Goal)
        return h
    def Compute_G(self,CurrentNode:StateNode,
                       NeighborNode:StateNode,SegmentLength=1.6):
        '''
        function: Calculate The G Cost
        '''
        if NeighborNode.direction == StateNode.FORWARD:
            if NeighborNode.steering_grade == CurrentNode.steering_grade:
                if NeighborNode.steering_grade == 0:
                    g = SegmentLength
                else:
                    g = SegmentLength * self.config['Steering']
            else:
                if NeighborNode.steering_grade == 0:
                    g = SegmentLength * self.config['SteeringChange']
                else:
                    g = SegmentLength * self.config['Steering'] * self.config['SteeringChange'] * (1 + (abs(NeighborNode.steering_grade) - 1) * 0.3)
        else:
            if NeighborNode.steering_grade == CurrentNode.steering_grade:
                if NeighborNode.steering_grade == 0:
                    g = SegmentLength * self.config['BackWard']
                else:
                    g = SegmentLength * self.config['Steering'] * self.config['BackWard'] * (1 + (abs(NeighborNode.steering_grade) - 1) * 0.3)
            else:
                if NeighborNode.steering_grade == 0:
                    g = SegmentLength * self.config['SteeringChange'] * self.config['BackWard']
                else:
                    g = SegmentLength * self.config['Steering'] * self.config['SteeringChange'] * self.config['BackWard'] * (1 + (abs(NeighborNode.steering_grade) - 1) * 0.3)
        if NeighborNode.direction != CurrentNode.direction:
            g *= self.config['DirectionChange']
        return g

    def ChooseStepLength(self,ExpandLength):
        self.ForwardSearch  = True
        self.BackwardSearch = True
        StepSize = 0.2
        self.ForwardStepLength  = np.clip(int(ExpandLength[0] * 0.6 / StepSize) * StepSize, 0.4 ,1.6)
        self.BackwardStepLength = np.clip(int(ExpandLength[2] * 0.6 / StepSize) * StepSize, 0.4 ,1.6)
        # self.ForwardStepLength  = 1.6
        # self.BackwardStepLength = 1.6

        # if ExpandLength[0] >= 2.0:
        #     self.ForwardStepLength = 1.6
        # elif ExpandLength[0] >= 1.4:
        #     self.ForwardStepLength = 1.0
        # elif ExpandLength[0] >= 1:
        #     self.ForwardStepLength = 0.6
        # elif ExpandLength[0] > 0.4:
        #     self.ForwardStepLength = 0.4
        # else:
        #     # if ExpandLength[0] < 0.2:
        #     #     self.ForwardSearch = False
        #     self.ForwardStepLength = 0.4
            

        # if ExpandLength[2] >= 2.0:
        #     self.BackwardStepLength = 1.6
        # elif ExpandLength[2] >= 1.4:
        #     self.BackwardStepLength = 1.0
        # elif ExpandLength[2] >= 1:
        #     self.BackwardStepLength = 0.6
        # elif ExpandLength[2] > 0.4:
        #     self.BackwardStepLength = 0.4
        # else:
        #     if ExpandLength[1] < 0.2:
        #         self.ForwardSearch = False
        #     self.BackwardStepLength = 0.4
        #     self.BackwardSearch = False
        # self.BackwardStepLength = 0.4
        # self.ForwardStepLength = 0.4
        # if ExpandLength[0] < 1 and ExpandLength[2] < 1: # 如果前后受限，即侧方情况
        #     self.DiscreteNum = 1
        # elif ExpandLength[1] < 0.3 and ExpandLength[3] < 0.3: # 如果左右受限，即前后情况
        #     self.DiscreteNum = 0
        # else:
        #     self.DiscreteNum = 2

        # self.DiscreteSteeringAngle = self.config['MaxSteering'] / DiscreteNum if DiscreteNum > 0 else 0
        
    def GetNeighbor(self,CurrentNode:StateNode,NeighborGroup:[])->[]:
        '''
        Func: Get the Neighbor State For Current State
        '''
        NeighborGroup.clear()
        x,y,yaw = CurrentNode.state
        if self.EmbodyMode:
            self.ExpandLength = self.GenerateCorridor(CurrentNode.state, MaxExpandLength=3)
            self.ChooseStepLength(self.ExpandLength)
        else:
            self.ForwardSearch,self.BackwardSearch = True,True
            self.ForwardStepLength,self.BackwardStepLength = 1.6,1.6
        # ForWard Simulate        
        for i in range(-self.DiscreteNum,self.DiscreteNum + 1):
            if self.ForwardSearch:
                Collision = False
                delta = i * self.DiscreteSteeringAngle
                TempState = self.GetMidState(self.ForwardStepLength,self.StepSize,delta,x,y,yaw)
                j = 0
                for state in TempState:
                    if not  self.CollisionCheck(state):
                        Collision = True
                        break
                    j += 1
                if not Collision:
                    StateGridIndex = self.State2Index(TempState[-1])
                    neighbor = StateNode(StateGridIndex)
                    neighbor.state = TempState[-1]
                    neighbor.direction = StateNode.FORWARD
                    neighbor.intermediate_states = TempState
                    neighbor.steering_grade = i
                    neighbor.SegmentLength = self.ForwardStepLength
                    neighbor.start_index = CurrentNode.start_index + len(neighbor.intermediate_states)
                    NeighborGroup.append(neighbor)
        # BackWard Simulation
            if self.BackwardSearch:
                Collision = False
                delta = i * self.DiscreteSteeringAngle
                TempState = self.GetMidState(-self.BackwardStepLength, -self.StepSize, delta, x, y, yaw)
                j = 0
                for state in TempState:
                    if not self.CollisionCheck(state):
                        Collision = True
                        break
                    j += 1
                if not Collision:
                    StateGridIndex = self.State2Index(TempState[-1])
                    neighbor = StateNode(StateGridIndex)
                    neighbor.state = TempState[-1]
                    neighbor.direction = StateNode.BACKWARD
                    neighbor.intermediate_states = TempState
                    neighbor.steering_grade = i
                    neighbor.SegmentLength = self.BackwardStepLength
                    neighbor.start_index = CurrentNode.start_index + len(neighbor.intermediate_states) # add path index
                    NeighborGroup.append(neighbor)
        return NeighborGroup
    def ChooseStart(self,Start:np.array,Goal:np.array):
        StartCorridor = self.GenerateCorridor(Start, MaxExpandLength=3)
        GoalCorridor = self.GenerateCorridor(Goal, MaxExpandLength=3)

        StartCorner = calculate_corners(Start[0], Start[1], Start[2], StartCorridor, self.veh.lf + self.veh.lw, self.veh.lr, self.veh.lb / 2)
        GoalCorner = calculate_corners(Goal[0], Goal[1], Goal[2], GoalCorridor, self.veh.lf + self.veh.lw, self.veh.lr, self.veh.lb / 2)
        # plt.plot(StartCorner[:, 0], StartCorner[:, 1], color='blue', label='Start Corners')
        # plt.plot(GoalCorner[:, 0], GoalCorner[:, 1], color='red', label='Goal Corners')
        print(f"StartArea: {calculate_polygon_area(StartCorner)},GoalArea: {calculate_polygon_area(GoalCorner)}")
        StartArea = calculate_polygon_area(StartCorner)
        GoalArea = calculate_polygon_area(GoalCorner)
        if GoalArea < StartArea:
            return Goal,Start
        else:
            return Start,Goal
        # return Start,Goal

    def Search(self,start:np.array,
                    goal:np.array,
                    StartIndex:int = 0,):
        '''
        Main Search Loop For Hybrid A Star
        '''
        self.Start = start
        self.Goal  = goal
        # self.Start,self.Goal = self.ChooseStart(start,goal)
        
        StartGridIndex = self.State2Index(self.Start)
        GoalGridIndex  = self.State2Index(self.Goal)

        StartNode = StateNode(StartGridIndex)
        StartNode.state = self.Start
        StartNode.steering_grade = 0
        StartNode.direction = StartNode.NO
        StartNode.node_status = StartNode.IN_OPENSET
        StartNode.intermediate_states.append(self.Start)
        StartNode.g_cost = 0
        StartNode.f_cost = self.Compute_H(StartNode)
        StartNode.start_index = StartIndex

        GoalNode  = StateNode(GoalGridIndex)
        GoalNode.state = self.Goal
        GoalNode.steering_grade = 0
        GoalNode.direction = StartNode.NO

        i,j,k = StartGridIndex
        self.closed_list[i][j][k] = StartNode
        i,j,k = GoalGridIndex
        self.closed_list[i][j][k] = GoalNode
        # 初始节点入栈
        heapq.heappush(self.open_list, (StartNode.f_cost, StartNode))
        Neighbor = []
        count = 0
        while self.open_list:
            _,CurrentNode = heapq.heappop(self.open_list)
            # plt.scatter(CurrentNode.state[0],CurrentNode.state[1])
            CurrentNode.node_status = StartNode.IN_CLOSESET
            if np.linalg.norm(CurrentNode.state[:2] - self.Goal[:2]) <= self.ShotDistance:
                rs_length = 0.0
                if self.AnalyticExpansion(CurrentNode, GoalNode, rs_length):
                    path,direction = self.get_path(GoalNode)
                    path = np.asarray(path)
                    direction = np.asarray(direction)
                    self.Reset()
                    return True, path,direction
            # 更新邻居节点
            Neighbor = self.GetNeighbor(CurrentNode,Neighbor)
            print("--------------Current Node--------------------")
            print(f"-------------Current State{CurrentNode.state}----------------------")
            print(f"f_cost:{CurrentNode.f_cost}    g_cost:{CurrentNode.g_cost}")
            print(f"------------------Cuurent Count:{count}-distance={np.linalg.norm(CurrentNode.state[:2] - self.Goal[:2])}--------------------------")
            for node in Neighbor:
                intermediate_states = np.array(node.intermediate_states)
                # plt.plot(intermediate_states[:,0],intermediate_states[:,1],color="black",linewidth=0.75,linestyle="-.")
                Neighbor_edge_cost = self.Compute_G(CurrentNode,node,SegmentLength=node.SegmentLength)
                Current_H = self.Compute_H(CurrentNode) * self.config['TieBreaker']
                i,j,k = node.grid_index
                if self.closed_list[i][j][k] is None:
                    node.g_cost = CurrentNode.g_cost + Neighbor_edge_cost
                    node.parent_node = CurrentNode
                    node.node_status = StartNode.IN_OPENSET
                    node.f_cost = node.g_cost + Current_H
                    heapq.heappush(self.open_list,(node.f_cost,node))
                    self.closed_list[i][j][k] = node
                elif self.closed_list[i][j][k].node_status == StateNode.IN_OPENSET:
                    g_temp = CurrentNode.g_cost + Neighbor_edge_cost
                    if g_temp < self.closed_list[i][j][k].g_cost:
                        node.g_cost = g_temp
                        node.parent_node = CurrentNode
                        node.node_status = StateNode.IN_OPENSET
                        node.f_cost = node.g_cost + Current_H
                        node.start_index = CurrentNode.start_index + len(node.intermediate_states)
                        self.closed_list[i][j][k] = node
                else:
                    continue
            count += 1
            # plt.pause(0.01)
            if count > 1e6:
                return False,[]
        return False,[]
    def AnalyticExpansion(self, CurrentNode, GoalNode, length) -> bool:
        feasible = self.ParkMap.TryRsPath(CurrentNode.state, GoalNode.state, self.config['StepSize'], length)
        if not feasible:
            return False
        rs_path_poses = self.ParkMap.GetRSPath(CurrentNode.state, GoalNode.state, self.config['StepSize'], length)
        GoalNode.intermediate_states = rs_path_poses[1:]
        GoalNode.parent_node = CurrentNode
        return True
    def get_path(self,GoalNode):
        '''
        Function: extract the Total Path Using BackTracking
        '''
        path,temp_nodes,direction = [],[],[]
        node = GoalNode
        while node is not None:
            temp_nodes.append(node)
            node = node.parent_node
        temp_nodes.reverse()
        for node in temp_nodes:
            path.extend(node.intermediate_states)
            direction.extend([node.direction] * len(node.intermediate_states))
        direction[0] = direction[1]
        direction[-1] = direction[-2]
        return path,direction


if __name__ == "__main__":
    plt.figure(dpi=300,figsize=(12,9))
    for i in range(1,120):
        if i == 7:
            continue
        res = 0.1
        # TPCAP_Case = Case(f'BenchmarkCases/Case{i}.csv', 0.01,res)
        TPCAP_Case = Case(f'LIOM_Benchmark/Case{i}.csv', 0.01,res)
        from map_test import MakeGridMap
        grid_binary = MakeGridMap(TPCAP_Case, grid_size=res)
        Cor = HybridAstar(72)
        Cor.Init(grid_binary, res, res, TPCAP_Case.xmax, TPCAP_Case.ymax, TPCAP_Case.xmin, TPCAP_Case.ymin)
        config = read_config("config")
        Hybrid_A_Star_planner = Hybrid_A_Star(Cor,config)

        Hybrid_A_Star_planner.ChooseStart(TPCAP_Case.GetStart(), TPCAP_Case.GetGoal())
        # TPCAP_Case.ShowMap()

        PlanningStatus,Path,Direction = Hybrid_A_Star_planner.Search(TPCAP_Case.GetStart(),TPCAP_Case.GetGoal(),StartIndex=0)
        # plt.imshow(grid_binary.T, 
        #         cmap='binary', 
        #         origin='lower',
        #             aspect='auto',
        #             extent=[TPCAP_Case.xmin, TPCAP_Case.xmax, TPCAP_Case.ymin, TPCAP_Case.ymax])
        # plt.scatter(TPCAP_Case.discrete_x, TPCAP_Case.discrete_y, color='blue', marker='o', label='Discrete Points')
        if PlanningStatus:
            
            # plt.plot(Path[:,0],Path[:,1])
            for point in Path:
                point = np.array(point)
                temp = Hybrid_A_Star_planner.veh.create_polygon(point[0], point[1], point[2])
                # plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.4, color='green')
            TotalCorridor = np.zeros((len(Path), 4))
            for j in range(len(Path)):
                TotalCorridor[j,:] = Hybrid_A_Star_planner.GenerateCorridor(Path[j], MaxExpandLength=5)
            for j in range(len(TotalCorridor)):
                conor = calculate_corners(Path[j][0], Path[j][1], Path[j][2], TotalCorridor[j])
                # plt.plot(conor[:, 0], conor[:, 1], linestyle='--', linewidth=0.4, color='red')
            # np.save(f'PlanningRes/LIOM_Case_{i}_Hybrid_A_star.npy', Path)
            np.save(f'PlanningRes/TPCAP_Case_{i}_Hybrid_A_star_Direction.npy', Direction)
            # np.save(f'PlanningRes/LIOM_Case_{i}_Corridor.npy', TotalCorridor)
        # TPCAP_Case.ShowMap(i=i,show=False)
        # padding = 5
        # plt.xlim(min(Path[:,0]) - padding,max(Path[:,0]) + padding)
        # plt.ylim(min(Path[:,1]) - padding,max(Path[:,1]) + padding)
        # plt.savefig(f"TPCAP_PDF/Case_{i}_Search.pdf")
        # plt.show()