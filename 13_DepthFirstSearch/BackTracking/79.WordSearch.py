class Solution:
    """
    Time: O(M*N)
    Space: O(M*N)
    """
    def exist(self, board: List[List[int]], word: str) -> bool:
        row, col = len(board), len(board[0])
        visited = [[False for _ in range(col)] for _ in range(row)]
        for i in range(row):
            for j in range(col):
                if self.backtrack(board, i, j, word, 0, visited):
                    return True
        return False

    def backtrack(self, board, i, j, word, index, visited):
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or visited[i][j] == True or board[i][j] != word[index]:
            return False
        if index == len(word) - 1 and board[i][j] == word[index]:
            return True
        visited[i][j] = True    # avoid visit agian 
        res = self.backtrack(board, i - 1, j, word, index + 1, visited) \
           or self.backtrack(board, i + 1, j, word, index + 1, visited) \
           or self.backtrack(board, i, j - 1, word, index + 1, visited) \
           or self.backtrack(board, i, j + 1, word, index + 1, visited)
        visited[i][j] = False
        return res


class Solution:
    """
    Time: O(M*N)
    Space: O(1)
    """
    def exist(self, board: List[List[int]], word: str) -> bool:
        row, col = len(board), len(board[0])
        for i in range(row):
            for j in range(col):
                if self.backtrack(board, i, j, word, 0):
                    return True
        return False

    def backtrack(self, board, i, j, word, index):
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] == "#" or board[i][j] != word[index]:
            return False
        if index == len(word) - 1 and board[i][j] == word[index]:
            return True
        temp = board[i][j]
        board[i][j] = "#"    # avoid visit agian
        res = self.backtrack(board, i - 1, j, word, index + 1) \
           or self.backtrack(board, i + 1, j, word, index + 1) \
           or self.backtrack(board, i, j - 1, word, index + 1) \
           or self.backtrack(board, i, j + 1, word, index + 1)
        board[i][j] = temp
        return res
