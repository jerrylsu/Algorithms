class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # end condition
        if not root:
            return 0
        # divide & conquer
        leftDepth = self.maxDepth(root.left)
        rightDepth = self.maxDepth(root.right)
        # combine：根节点的最大深度 = max(左子树最大深度，右子树最大深度) + 1
        res = max(leftDepth, rightDepth) + 1
        # return result
        return res
