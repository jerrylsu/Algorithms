class Solution:
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        row, col = len(heights), len(heights[0])
        pa_status = [[False for _ in range(col)] for _ in range(row)]
        al_status = [[False for _ in range(col)] for _ in range(row)]
        
        for r in range(row):
            self.dfs(heights, r, 0, pa_status)
            self.dfs(heights, r, col-1, al_status)
        
        for c in range(col):
            self.dfs(heights, 0, c, pa_status)
            self.dfs(heights, row-1, c, al_status)
        
        results = []
        for i in range(row):
            for j in range(col):
                if pa_status[i][j] == True and al_status[i][j] == True:
                    results.append([i, j])
        return results
    
    def dfs(self, heights, r, c, status):
        if status[r][c] == True:
            return
        status[r][c] =True
        for direction in self.directions:
            r_new, c_new = r + direction[0], c + direction[1]
            if r_new >= 0 and r_new < len(heights) and c_new >= 0 and c_new < len(heights[0]) and heights[r_new][c_new] >= heights[r][c]:
                self.dfs(heights, r_new, c_new, status)
