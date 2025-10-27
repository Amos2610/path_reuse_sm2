from xarm_utils_py import XArmUtils, Node


class XArmUtilsWrapper:
    """任意スキルから継承して xarm をそのまま使える薄い基底クラス"""
    
    xarm_node = None
    _xarm = None
    _node_name = "xarm_core_node"
    _arm_name = "xarm6"

    def __init__(self):
        if XArmUtilsWrapper.xarm_node is None:
            XArmUtilsWrapper.xarm_node = Node(XArmUtilsWrapper._node_name)
        if XArmUtilsWrapper._xarm is None:
            XArmUtilsWrapper._xarm = XArmUtils(XArmUtilsWrapper.xarm_node, XArmUtilsWrapper._arm_name)

        # インスタンスから参照できるように設定
        self.xarm_node = XArmUtilsWrapper.xarm_node
        self.xarm = XArmUtilsWrapper._xarm

    def set_pipeline(self, name: str):
        self.xarm.set_planning_pipeline(name)
