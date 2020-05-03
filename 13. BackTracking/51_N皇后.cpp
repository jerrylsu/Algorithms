#include<iostream>
#include<vector>
#include<string>

using namespace std;

class Solution{
    public:
        vector<vector<string>> solveNQueens(int n){
            vector<vector<string>> res;
            vector<string> nQueens(n, string(n, '.'));
            dfs(nQueens, res, 0, n);
            return res;
        }

        void dfs(vector<string>& nQueens, vector<vector<string>>& res, int row, int& n){
            if(row == n){                             // N层树
                res.push_back(nQueens);
                return;
            }
            for(int col = 0; col < n; ++col){         // N叉树
                if(isValid(nQueens, row, col, n)){    // 剪枝
                    nQueens[row][col] = 'Q';
                    dfs(nQueens, res, row + 1, n);
                    nQueens[row][col] = '.';
                }
            }
        }

        bool isValid(vector<string>& nQueens, int row, int col, int& n){
            // 检查坐标(row, col)在column（上半部）上是否存在'Q'
            for(int i = 0; i < row; ++i){
                if(nQueens[i][col] == 'Q') return false;
            }
            // 检查坐标(row, col)在45°对角线（上半部）上是否存在'Q'
            for(int i = row, j = col; i >= 0 && j < n; --i, ++j){
                if(nQueens[i][j] == 'Q') return false;
            }
            // 检查坐标(row, col)在135°对角线（上半部）上是否存在'Q'
            for(int i = row, j = col; i >= 0 && j >= 0; --i, --j){
                if(nQueens[i][j] == 'Q') return false;
            }
            return true;
        }
};

class Solution_error{
    /* 第一次写的代码，存在很多问题：
    * 1. 按行的每一个位置进行回溯的思路是正确的。
    * 2. 需要设置row, col, 45°, 135°线上的攻击标志吗？
    */
    private:
        vector<vector<string>> res;
        vector<string> track;
    public:
        vector<vector<string>> solveNQueens(int n){
            if(n < 1) return res;
            vector<vector<bool>> visited(n, vector<bool>(n, false));
            vector<string> s(n, string(n, '.'));
            dfs(res, s, visited, 0, n);
            return res;
        }

        void dfs(vector<vector<string>>& res, vector<string>& s, vector<vector<bool>>& visited, int level, int n){
            if(level == n){
                res.push_back(track);
                return;
            }
            for(int col = 0; col < n; ++col){
                if(!visited[level][col]){
                    set_visited(visited, level, col, true);
                    s[level][col] = 'Q';
                    track.push_back(s[level]);
                    dfs(res, s, visited, level + 1, n);
                    s[level][col] = '.';
                    track.pop_back();
                    set_visited(visited, level, col, false);
                }
            }
        }

        void set_visited(vector<vector<bool>>& visited, int x, int y, bool flag){
            int row = visited.size();
            int col = visited[0].size();

            for(int i = 0; i < col; ++i) visited[x][i] = flag; // 需要删除，行是不需要检查的，不会有冲突！！
            for(int i = 0; i < row; ++i) visited[i][y] = flag;
            for(int i = 0; i < row; ++i){
                for(int j = 0; j <col; ++j){
                    if((x - y) == (i - j) || (x + y) == (i + j)){
                        visited[i][j] = flag;
                    }
                }
            }
        }

        void show_visited(vector<vector<bool>>& visited){
            int row = visited.size();
            int col = visited[0].size();
            for(int i = 0; i < row; ++i){
                for(int j = 0; j < col; ++j) cout << visited[i][j] << " ";
                cout << endl;
            }
        }
};

void test_set_visited(){
    Solution_error solution;
    vector<vector<bool>> visited(8, vector<bool>(8, false));
    solution.show_visited(visited);
    cout << endl;
    solution.set_visited(visited, 2, 2, true);  // 将坐标(i, j)的攻击范围设置为true
    solution.show_visited(visited);
    cout << endl;
    solution.set_visited(visited, 2, 2, false); // 将坐标(i, j)的攻击设置为false
    solution.show_visited(visited);
}

int main(){
    //test_set_visited();
    Solution solution;
    vector<vector<string>> res;
    res = solution.solveNQueens(4);
    for(auto& ele : res){
        for(auto& s : ele) cout << s << endl;
    }
}