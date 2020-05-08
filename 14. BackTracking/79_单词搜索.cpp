#include<vector>
#include<iostream>

using std::vector;
using std::string;

class Solution {
    private:
        int row, col;
        const int delta[4][2]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; // {}列表元素拷贝初始化
    public:
        bool exist(vector<vector<char>>& board, string word) {
            if(board.empty() || word.empty()){
                return false;
            }
            row = board.size();
            col = board[0].size();
            vector<vector<bool>> visted(row, vector<bool>(col, false)); // 构造器初始化(num, content)
            for(int i = 0; i < row; i++){
                for(int j = 0; j < col; j++){
                    // 如果搜到直接返回，否则继续搜索
                    if(dfs(i, j, board, word, visted, 0)){ // true立即返回，false继续搜索
                        return true;
                    }
                }
            }
            return false;
        }

        bool dfs(int x, \
                 int y, \
                 vector<vector<char>>& board, \
                 string word, \
                 vector<vector<bool>>& visted, \
                 int index){
            std::cout << "index " << index << ": " << board[x][y] << "->" << word.at(index) << "\n";
            if(index == word.size()-1){
                return board[x][y] == word.at(index);
            }
            if(board[x][y] == word.at(index)){
                visted[x][y] = true;
                // 分别从四个方向进行搜索
                for(int i = 0; i < 4; i++){
                    int newRow = x + delta[i][0];
                    int newCol = y + delta[i][1];
                    std::cout << "(" << newRow << ", " << newCol << ")\n";
                    if(checkValid(newRow, newCol, visted) && dfs(newRow, newCol, board, word, visted, index + 1)){
                        return true;
                    }
                }
                // 当前点(x, y)的四个方向都没搜到，回溯需要重置visted[x][y]为false，用于其他位置开始查询。
                visted[x][y] = false;
            }
            return false;
        }

        bool checkValid(int x, int y, vector<vector<bool>>& visted){
            if(x >= 0 && x < row && y >= 0 && y < col){
                return visted[x][y] == false;
            }
            return false;
        }
};

int main(){
    vector<vector<char>> board = {{'A','B','C','E'}, {'S','F','C','S'}, {'A','D','E','E'}};
    vector<vector<bool>> visted(board.size(), vector<bool>(board[0].size(), false));
    for(int i = 0; i < board.size(); i++){
        for(int j = 0; j < board[0].size(); j++){
            std::cout << board[i][j] << " ";
        }
        std::cout << "\n";
    }

    Solution s;
    if(s.exist(board, "ABCCED")){
        std::cout << "yes" << "\n";
    }else{
        std::cout << "no" << "\n";
    }
    
    return 0;
}