'''
@Project :PandaParking
@Author : YuhaoDu
@Date : 2025/4/5 
'''
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from Optimize_util import *
from TPCAP_Cases import calculate_corners
import time

class EmbodySmoother:
    def __init__(self, ref_p,
                        corridor,
                        init_control=None,
                        init_Tf = None):
        '''
        func: smooth the raw path
        :param ref_p:the reference path point
        '''
        ref_p[:,3] = Checkyaw(ref_p[:,3])
        # 参考轨迹
        self.init_p     = ref_p
        self.ref_p      = ref_p
        self.ref_control = init_control if init_control is not None else np.zeros((len(ref_p)-1,2))
        self.Corridor = corridor
        self.N          = len(self.ref_p)
        # 共有5个状态量
        self.n_states   = 5
        # 变量约束
        self.lbx        = []
        self.ubx        = []
        # 变量
        self.variable   = []
        # 约束
        self.constrains = []
        # 约束条件上下限
        self.lbg        = []
        self.ubg        = []
        # 初始解
        self.x0         = []
        self.start_pose = ref_p[0]
        self.end_pose   = ref_p[-1]
        # 固定时间窗 控制量  [acc, delta_f]
        # self.n_controls = 2
        # 可变时间窗 控制量修改为[acc, delta_f, t]
        self.n_controls = 2
        self.vio_coe = 1e2
        self.T_max = 1
        self.r = Matrix_R
        self.q = Matrix_Q
        self.SolveTime = 0
        self.DEBUG = False
        if init_Tf is not None:
            self.TF_opt = init_Tf

        self.traj_hist = []
        self.control_hist = []
        self.time_hist = []

    def initialize(self):

        self.x0 = []


        self.init_state = ca.SX([   self.ref_p[0][0],
                                    self.ref_p[0][1],
                                    self.ref_p[0][2],
                                    self.ref_p[0][3],
                                    self.ref_p[0][4]])
        # 终止状态约束
        self.end_state = ca.SX([self.ref_p[-1][0],
                                self.ref_p[-1][1],
                                self.ref_p[-1][2],
                                self.ref_p[-1][3],
                                self.ref_p[-1][4]])
        for i in range(0,len(self.ref_p)):
            # x , y  , v , yaw, a ,steer
            self.x0 += [self.ref_p[i][0] ,self.ref_p[i][1], self.ref_p[i][2], self.ref_p[i][3],self.ref_p[i][4]]
        # 增加控制量
        if len(self.traj_hist) > 0:
            for i in range(self.x_opt.shape[0] - 1):
                self.x0 += [self.a_opt[i],self.steerate_opt[i]]
        elif self.ref_control is not None:
            self.x0 += [self.ref_control[i] for i in range(self.ref_control.shape[0])]
        else:
            self.x0 += [[0]*(self.n_controls*(self.N-1))]
            # for i in range(len(self.ref_p) - 1):
            #     self.x0 += [self.ref_control[i]]
        self.x0 += [self.TF_opt] if hasattr(self,'TF_opt') else [self.N] 
    def build_model(self):
        '''
        func:build the nolinear model and construct the variable
        :return:
        '''
        # 构造符号变量
        x       = ca.SX.sym('x')
        y       = ca.SX.sym('y')
        theta   = ca.SX.sym('theta')
        v       = ca.SX.sym('v')
        steering= ca.SX.sym('steering')
        a       = ca.SX.sym('a')
        t_f       = ca.SX.sym('t')
        # 步长t
        steering_rate = ca.SX.sym('steering_rate')
        self.GetCorner = ca.Function("GetCorner",[x,y,theta],[CalVehicleCorner(x,y,theta)])
        self.obj = 0

        # state:[x,y,v,\theta, steering]
        self.state      = ca.vertcat(x, y, v, theta, steering)
        # control:[acc, delta_f]
        self.control    = ca.vertcat(a, steering_rate)

        self.rhs = ca.vertcat(v * ca.cos(theta),
                              v * ca.sin(theta),
                              a,
                              v / C.lw * ca.tan(steering),
                              steering_rate)
        # 构建f = AX+BU
        self.f = ca.Function('f', [self.state, self.control], [self.rhs])
        # 构建状态量集合 N个时刻 每个时刻5个状态量
        self.X = ca.SX.sym('X', self.n_states, self.N)
        # 构建控制量集合 N-1个间隔， 每个时刻2个控制量
        self.U = ca.SX.sym('U', self.n_controls, self.N - 1)
        self.TF = ca.SX.sym('TF')


    def generate_obj(self):
        '''
        func: generate the objective func of path smoother
        :return:
        '''
        R = ca.SX(self.r)
        Q = ca.SX(self.q)
        self.obj  = self.TF
        dt = self.TF/(self.N - 1)

    def generate_variable(self):
        '''
        func: generate the raw variable to be optimized
        :return:
        '''
        self.variable = []
        self.lbx = []
        self.ubx = []
        for i in range(self.N):
            # 一阶控制约束
            self.variable += [self.X[:, i]]
            self.lbx += [-np.inf, -np.inf, - C.MAX_SPEED,  -40 * np.pi , -C.MAX_STEER]
            self.ubx += [ np.inf,  np.inf,   C.MAX_SPEED,   40 * np.pi ,  C.MAX_STEER]

        for i in range(self.N - 1):
            # 二阶控制约束
            self.variable += [self.U[:, i]]
            self.lbx += [-C.MAX_ACC, -C.MAX_STEERING_RATE ]
            self.ubx += [ C.MAX_ACC,  C.MAX_STEERING_RATE ]
        self.variable += [self.TF]
        self.lbx += [0]
        self.ubx += [self.N * self.T_max]

    def generate_constraint(self):
        '''
        func: generative the constraint of smooth term
        :return:
        todo: replace the nonlinear curvature constraint with linear constraint
        '''
        self.vio = 0
        # 增加起始位置约束
        self.constrains += [self.X[:, 0] - self.ref_p[0][:5]]
        self.lbg += [0, 0, 0, 0, 0]
        self.ubg += [0, 0, 0, 0, 0]
        # 增加过程中可行约束
        for i in range(self.N - 1):
            st = self.X[:, i]
            st_next = self.X[:,i+1]
            u = self.U[:,i]
            dt = self.TF/(self.N - 1)

            f_value = self.f(st, u) * dt
            self.vio += ((st_next - st) - f_value).T @ ((st_next - st) - f_value) 
            
            # self.constrains += [st_next - (st + fdt)]
            # self.lbg += [0,0,0,0,0]
            # self.ubg += [0,0,0,0,0]

            corner = self.GetCorner(st[0],st[1],st[3])
            a_1,b_1,c_1_max,c_1_min,a_2,b_2,c_2_max,c_2_min = self.Corridor[i] #a1,b1,c1_max,c1_min,a2,b2,c2_max,c2_min
            for j in range(corner.shape[0]):
                x,y = corner[j], corner[4+j]
                val_1 = -a_1 * x - b_1 * y
                val_2 = -a_2 * x - b_2 * y
                self.constrains += [val_1,
                                    val_2] #  y-kx  (y = kx + b) ==> (y - kx = b)
                self.lbg += [c_1_min, c_2_min]
                self.ubg += [c_1_max, c_2_max]

        # 增加结束状态约束
        self.constrains += [self.X[:, -1] - self.ref_p[-1][:5]]
        self.lbg += [0, 0, 0, 0, 0]
        self.ubg += [0, 0, 0, 0, 0]

    def solve(self):
        '''
        solve the nonlinear problem
        :return:
        '''
        if len(self.control_hist) == 0:
            self.initialize()
            self.build_model()
            self.generate_obj()
            self.generate_variable()
            self.generate_constraint()
            self.obj += self.vio * self.vio_coe
        else:
            self.constrains = []
            self.lbg = []
            self.ubg = []
            self.initialize()
            self.generate_obj()
            self.generate_variable()
            self.generate_constraint()
            # self.T_max *= 10
            self.obj += self.vio * self.vio_coe
        self.cal_vio = ca.Function("cal_vio", [ca.vertcat(*self.variable)], [self.vio])
        # plt.plot(self.ref_p[:, 0], self.ref_p[:, 1])
        # plt.show()
        print("---------------------Solving-------------------")
        nlp_prob = {'f': self.obj, 'x': ca.vertcat(*self.variable),
                    'g': ca.vertcat(*self.constrains)}
        opts_setting = {'ipopt.max_iter': 500,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-3,
                        'ipopt.acceptable_obj_change_tol': 1e-3,
                        "jit":False,
                        "verbose" :False,
                        "ipopt.mumps_mem_percent" : 500 ,
                        # 'ipopt.mu_strategy': 'adaptive',  # 自适应障碍参数更新
                        'ipopt.fast_step_computation': 'yes',  # 加速步长计算
                        # 'ipopt.hessian_approximation': 'limited-memory',  # 对大规模问题更高效
                        # 'iteration_callback': self.SolveCallBack
                        }
        # 构造求解器 选择求解其为ipopt
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob,opts_setting)
        # solver = ca.nlpsol('solver', 'ipopt', nlp_prob)

        sol = solver(x0=ca.vertcat(*self.x0), 
                     lbx=ca.vertcat(*self.lbx), 
                     ubx=ca.vertcat(*self.ubx),
                     ubg=ca.vertcat(self.ubg), 
                     lbg=ca.vertcat(self.lbg))

        res = sol['x']
        self.sol = sol
        self.x_opt = res[0:self.n_states * (self.N):self.n_states]
        self.y_opt = res[1:self.n_states * (self.N):self.n_states]
        self.v_opt = res[2:self.n_states * (self.N):self.n_states]
        self.theta_opt = res[3:self.n_states * (self.N):self.n_states]
        self.steer_opt = res[4:self.n_states * (self.N):self.n_states]
        self.a_opt =        res[self.n_states * (self.N):self.n_states * (self.N) + self.n_controls * (self.N - 1):self.n_controls]
        self.steerate_opt   = res[self.n_states * (self.N) + 1:self.n_states * (self.N) + self.n_controls * (self.N - 1):self.n_controls]
        self.TF_opt          = res[-1]
        # print(self.t)
        self.t_cum = np.arange(self.N) * (self.TF_opt/self.N)
        self.t_opt = np.arange(self.N) * (self.TF_opt/self.N)
        self.traj_hist.append(np.hstack((self.x_opt,
                                         self.y_opt,
                                         self.v_opt,
                                         self.theta_opt,
                                         self.steer_opt)))
        self.control_hist.append(np.hstack((self.a_opt,
                                            self.steerate_opt)))
        self.time_hist.append(self.TF_opt)
        self.res = sol['x']
        return res
    def IterativeSolve(self, Cor,max_iter=5):
        for i in range(max_iter):
            path = self.ref_p[:,[0,1,3]]
            self.Corridor = MakeCorridor(path, Cor)
            StartTime = time.time()
            self.solve()
            self.SolveTime += time.time() - StartTime
            # self.vio_coe *= 1e2
            vio = ca.sqrt(self.cal_vio(self.res)/self.vio_coe)
            if i > 0:
                PathNorm = np.linalg.norm(self.ref_p[:, :] - self.traj_hist[-1][:])
                self.ref_p = self.traj_hist[-1]
                Coverage = True if PathNorm / len(self.ref_p) < 1e-1  else False
            else :
                Coverage = False
            if vio < 1e-2 and Coverage:
                # self.vio_coe *= 10
                print(f"Converged at iteration {i+1} with violation {vio}")
                break
            else:
                print(f"Iteration {i+1} with violation {vio}")
                if vio < 1e-1:
                    self.vio_coe = min(self.vio_coe*2, 1e6)
                else:
                    self.vio_coe = max(self.vio_coe*5, 1e6)
    
def GetHalfSpace(ExpandLength,Point):
    Corners = calculate_corners(Point[0], Point[1], Point[2], ExpandLength)
    HalfSpace = np.zeros((4,3))
    for i in range(4):
        point_0 = Corners[i]
        point_1 = Corners[(i+1)%4]

        a = point_1[1] - point_0[1]
        b = point_0[0] - point_1[0]
        c = point_1[0]*point_0[1] - point_0[0]*point_1[1]

        HalfSpace[i,:] = [a, b, c]
    HalfSpace[0] = -HalfSpace[0]
    HalfSpace[1] = -HalfSpace[1]
    a_1 = HalfSpace[0][0]
    b_1 = HalfSpace[0][1]
    c_1_max = max(HalfSpace[0][2],HalfSpace[2][2])
    c_1_min = min(HalfSpace[0][2],HalfSpace[2][2])

    a_2 = HalfSpace[1][0]
    b_2 = HalfSpace[1][1]
    c_2_max = max(HalfSpace[1][2],HalfSpace[3][2])
    c_2_min = min(HalfSpace[1][2],HalfSpace[3][2])

    return [a_1,b_1,c_1_max,c_1_min,a_2,b_2,c_2_max,c_2_min]
def MakeCorridor(Path, Hybrid_A_Star_planner):
    MaxExpandLength = 7.5
    delta_s = 0.1
    ExpandLength = [0]*4
    TotalCorridor = np.zeros((len(Path), 4))
    HalfSpaceConstraint = []
    for j in range(len(Path)):
        TotalCorridor[j,:] = Hybrid_A_Star_planner.GenerateCorridor(Path[j], MaxExpandLength,delta_s,ExpandLength)
    for j in range(len(TotalCorridor)):
        corner = calculate_corners(Path[j][0], Path[j][1], Path[j][2], TotalCorridor[j])
    for pose,LocalCorridor in zip(Path,TotalCorridor):
            HalfSpace = GetHalfSpace(LocalCorridor,pose)
            HalfSpaceConstraint.append(HalfSpace)
    return HalfSpaceConstraint

if __name__ == '__main__':
    from KinematicModel import Vehicle
    from map_test import MakeGridMap
    from HybridAstar import HybridAstar
    from TPCAP_Cases import Case    
    from Optimize_util import downsample_trajectory
    veh = Vehicle()
    Save = False
    show = True
    benchmark = "TPCAP"
    for i in range(18,120):
        print(f"Processing Case {i}")
        res = 0.1
        if benchmark == "LIOM":
            TPCAP_Case = Case(f'LIOM_Benchmark/Case{i}.csv', 0.01,res)
            PlanningResult = np.load(f'PlanningRes/LIOM_Case_{i}_Hybrid_A_star.npy')
        elif benchmark == "TPCAP":
            if i ==7: continue
            TPCAP_Case = Case(f'BenchmarkCases/Case{i}.csv', 0.01,res)
            PlanningResult = np.load(f'PlanningRes/TPCAP_Case_{i}_Hybrid_A_star.npy')
        else:
            raise ValueError("没有这个选项")
        dist_1 = np.linalg.norm(TPCAP_Case.GetGoal()[:-1] - PlanningResult[0,:2])
        dist_2 = np.linalg.norm(TPCAP_Case.GetStart()[:-1] - PlanningResult[0,:2])

        reversed = False if dist_1 > dist_2 else True
        PlanningResult = downsample_trajectory(PlanningResult, target_num=100)
        if reversed:
            PlanningResult = PlanningResult[::-1]
        from down_sample import down_sample
        init_res,init_control, Tf = down_sample(PlanningResult)
        grid_binary = MakeGridMap(TPCAP_Case, grid_size=res)
        Cor = HybridAstar(72)
        Cor.Init(grid_binary, res, res, TPCAP_Case.xmax, TPCAP_Case.ymax, TPCAP_Case.xmin, TPCAP_Case.ymin)
        # path = np.vstack((PlanningResult[:,0],
        #                   PlanningResult[:,1],
        #                   np.zeros(PlanningResult.shape[0]),
        #                   PlanningResult[:,2],
        #                   np.zeros(PlanningResult.shape[0]),
        #                   )).T
        # control = np.vstack((np.zeros(PlanningResult.shape[0] - 1), np.zeros(PlanningResult.shape[0] - 1), np.zeros(PlanningResult.shape[0] - 1))).T
        # control = np.vstack((a[:-1], np.zeros(PlanningResult.shape[0] - 1), t[:-1])).T
        HalfSpaceConstraint = MakeCorridor(PlanningResult, Cor)
        # Smoother = EmbodySmoother(path, control,HalfSpaceConstraint)
        Smoother = EmbodySmoother(init_res, HalfSpaceConstraint,None)

        # Smoother = RadauEmbodySmoother(path, HalfSpaceConstraint,Nfe=25)
        start = time.time()
        Smoother.IterativeSolve(Cor)
        # Smoother.solve()
        end = time.time()
        print(f"Iterative Solve Time: {end - start:.2f} seconds")
        PlanningRes = np.hstack((Smoother.x_opt,Smoother.y_opt,Smoother.theta_opt,Smoother.v_opt,Smoother.steer_opt))
        # ContorlRes = np.hstack((Smoother.a_opt,Smoother.steerate_opt,Smoother.t_cum[:-1]))
        ContorlRes = Smoother.t_opt[-1].toarray()
        if Save:
            if benchmark == "TPCAP":
                np.save(f'Smooth_Res/IterativeSolveEmbody/TPCAP_Case_{i}_Iterative_Embody_Smoother.npy', PlanningRes)
                np.save(f'Smooth_Res/IterativeSolveEmbody/TPCAP_Case_{i}_Iterative_Embody_Smoother_hist.npy', np.array(Smoother.traj_hist))
            else:
                np.save(f'Smooth_Res/IterativeSolveEmbody/LIOM_Case_{i}_Iterative_Embody_Smoother.npy', PlanningRes)
                np.save(f'Smooth_Res/IterativeSolveEmbody/LIOM_Case_{i}_Iterative_Embody_Smoother_hist.npy', np.array(Smoother.traj_hist))
        # for pose in PlanningRes:
        #     temp = veh.create_polygon(pose[0], pose[1], pose[2])
        #     plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth=0.4, color='blue')
        # plt.plot(PlanningResult[:,0], PlanningResult[:,1])
        # for i in range(len(Smoother.traj_hist)):
        #     plt.plot(Smoother.traj_hist[i][:,0], Smoother.traj_hist[i][:,1], linewidth=0.5,label=f"iteration {i+1}")
        # # plt.plot(PlanningRes[:,0], PlanningRes[:,1])
        # plt.plot(Smoother.x_opt,Smoother.y_opt,color="red",label="optimized")
        # TPCAP_Case.ShowMap(i,show=True)

        TPCAP_Case.ShowRes(PlanningResult,PlanningRes,ContorlRes,i,show = show,save=Save)