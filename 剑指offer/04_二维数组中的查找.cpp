#include<vector>

class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if(matrix.size() == 0) return false;
        int row = matrix.size() - 1, col = 0;
        while(row >= 0 && col < matrix[0].size()) {
            if(matrix[row][col] > target) {
                row -= 1;
            }
            else if(matrix[row][col] < target) {
                col += 1;
            }
            else {
                return true;
            }
        }
        return false;
    }
};
