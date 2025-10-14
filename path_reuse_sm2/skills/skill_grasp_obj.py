# from yasmin.blackboard import Blackboard
# from path_reuse_sm2.core.base import Skill
# from path_reuse_sm2.core.plugin import skill

# @skill(names=["SkillGraspObj", "GraspObj"])
# class SkillGraspObj(Skill):
#     def run(self, *, bb: Blackboard, **kwargs) -> bool:
#         self.node.get_logger().info("[SkillGraspObj] Grasp (dummy)")
#         return True

# path_reuse_sm2/skills/skill_grasp_obj.py
from path_reuse_sm2.core.state import SkillState
from path_reuse_sm2.core.plugin import skill  # 既存の自動登録デコレータ

@skill("SkillGraspObj")
class SkillGraspObj(SkillState):
    def __init__(self, node, **kwargs):
        super().__init__(node, name="SkillGraspObj")

    def run(self, bb, **kwargs) -> bool:
        self.node.get_logger().info("[SkillGraspObj] Grasp (dummy)")
        # ここに本処理（成功=True / 失敗=False）
        return True
