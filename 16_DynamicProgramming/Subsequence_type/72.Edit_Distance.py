class Solution:
    """
    """
    def minDistance(self, word1: str, word2: str) -> int:
        """Recursion.Top to Bottom.
            ed(m-1, n): delete,
            ed(m-1, n-1): replace,
            ed(m, n-1): insert, m+1-1， n-1
        """
        def ed(m, n):
            if m == -1:
                return n + 1
            if n == -1:
                return m + 1
            if word1[m] == word2[n]:
                return ed(m-1, n-1)
            else:
                return min(ed(m-1, n), ed(m-1, n-1), ed(m, n-1)) + 1
        m, n = len(word1) - 1, len(word2) - 1
        return ed(m, n)

    def minDistance(self, word1: str, word2: str) -> int:
        """Recursion wit memory. Top to Bottom.
        """
        mem = {}
        def ed(m, n):
            if (m, n) in mem:
                return mem[(m, n)]
            if m == -1:
                return n + 1
            if n == -1:
                return m + 1
            if word1[m] == word2[n]:
                mem[(m, n)] = ed(m-1, n-1)
            else:
                mem[(m, n)] = min(ed(m-1, n), ed(m-1, n-1), ed(m, n-1)) + 1
            return mem[(m, n)]
        m, n = len(word1) - 1, len(word2) - 1
        return ed(m, n)
