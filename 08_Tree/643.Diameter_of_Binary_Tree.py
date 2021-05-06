class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.diameter = 0 
        self.helper(root)
        return self.diameter
        
    def helper(self, root):
        if not root:
            return 0
        left = self.helper(root.left)
        right = self.helper(root.right)
        self.diameter = max(self.diameter, left + right)
        return 1 + max(left, right)
