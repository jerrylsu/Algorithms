class Solution:
    """O(M*N)：所有节点只遍历一次。
    M*N个节点；每个节点有4条边。
    """
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        if not grid: return 0
        row, col = len(grid), len(grid[0])
        max_area = 0
        for i in range(row):
            for j in range(col):
                max_area = max(max_area, self.dfs(grid, i, j))
        return max_area
    
    def dfs(self, grid, i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
            return 0
        grid[i][j] = 0  # do choice, mark visited
        return 1 + self.dfs(grid, i - 1, j) + self.dfs(grid, i + 1, j) + self.dfs(grid, i, j - 1) + self.dfs(grid, i, j + 1)
