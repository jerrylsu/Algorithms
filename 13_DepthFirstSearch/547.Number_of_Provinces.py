class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        node = len(isConnected)
        visited = [False] * node
        count = 0
        for i in range(node):
            if visited[i] == False:
                self.dfs(isConnected, i, visited)
                count += 1
        return count
    
    def dfs(self, isConnected, i, visited):
        visited[i] = True
        for j in range(len(isConnected)):
            if isConnected[i][j] == 1 and visited[j] == False:
                self.dfs(isConnected, j, visited)
