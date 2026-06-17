class StateNode:

    NOT_VISITED = 0
    IN_OPENSET = 1
    IN_CLOSESET = 2


    FORWARD = 0
    BACKWARD = 1
    NO = 3

    def __init__(self, grid_index):
        """
        :param grid_index: 一个包含三个整数的元组或列表，例如 (i, j, k)
        """
        self.node_status = StateNode.NOT_VISITED  # 初始未访问
        self.direction = StateNode.NO             # 默认方向为 NO
        self.state = None                         # 期望为一个 NumPy 数组 [x, y, theta]
        self.grid_index = grid_index              # 离散化后的索引 (i, j, k)
        self.g_cost = 0.0                         # 从起点到该节点的代价
        self.f_cost = 0.0                         # 总代价（g_cost + h_cost）
        self.steering_grade = 0                   # 转向等级
        self.parent_node = None                   # 父节点引用
        self.intermediate_states = []             # 中间状态列表，每个状态期望为 NumPy 数组 [x, y, theta]
        self.SegmentLength = 0.4

    def reset(self):
        """
        重置节点状态和父节点引用，用于后续重新搜索。
        """
        self.node_status = StateNode.NOT_VISITED
        self.parent_node = None

    def __lt__(self, other):
        '''
        revise compare function for PriorityQueue
        '''
        result = False
        if self.f_cost < other.f_cost:
            result = True
        return result