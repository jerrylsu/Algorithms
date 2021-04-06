#include<string>
#include<vector>
#include<algorithm>
#include<iostream>

class Solution {
    private:
        std::vector<std::string> res;
        std::string track;
    public:
        std::vector<std::string> permutation(std::string s) {
            if(s.empty()){
                return res;
            }
            std::vector<bool> visited(s.size(), false);
            std::sort(s.begin(), s.end());
            dfs(res, track, s, visited);
            return res;
        }

        void dfs(std::vector<std::string>& res, std::string& track, std::string& s, std::vector<bool>& visited){
            if(track.size() == s.size()){
                res.push_back(track);
            }

            for(int i = 0; i < s.size(); i++){
                // if(visited[i]){
                //     continue;
                // }

                // if(i > 0 && visited[i-1] && s[i-1] == s[i]){
                //     continue;
                // }
                if(!visited[i] && i<=0 && !visited[i] && s[i-1] != s[i]){
                    visited[i] = true;
                    track.push_back(s[i]);
                    dfs(res, track, s, visited);
                    track.pop_back();
                    visited[i] = false;
                }
            }
        }
};

int main(){
    std::string str("abc");
    std::vector<std::string> res;
    Solution s;
    res = s.permutation(str);

    for(int i = 0; i < res.size(); i++){
        std::cout << res[i] << " ";
    }
    std::cout << "\n";
    return 0;
}