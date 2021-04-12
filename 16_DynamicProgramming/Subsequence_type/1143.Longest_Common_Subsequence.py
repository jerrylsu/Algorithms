class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """Recursion.
        """
        def lcs(m, n):
            if m == -1 or n == -1:
                return 0
            if text1[m] == text2[n]:
                return lcs(m-1, n-1) + 1
            else:
                return max(lcs(m-1, n), lcs(m, n-1))
        m, n = len(text1) - 1, len(text2) - 1
        return lcs(m, n)

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """Recursion with memory.
        """
        mem = {}
        def lcs(m, n):
            if (m, n) in mem:
                return mem[(m, n)]
            if m == -1 or n == -1:
                return 0
            if text1[m] == text2[n]:
                mem[(m, n)] = lcs(m-1, n-1) + 1
            else:
                mem[(m, n)] = max(lcs(m-1, n), lcs(m, n-1))
            return mem[(m, n)]
        m, n = len(text1) - 1, len(text2) - 1
        return lcs(m, n)
