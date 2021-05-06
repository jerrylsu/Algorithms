class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        return self.helper(root) != -1
        
    def helper(self, root):
        if not root:
            return 0
        left = self.helper(root.left)
        if left == -1:
            return -1
        right = self.helper(root.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return 1 + max(left, right)
