#include<iostream>
#include<string>
#include<vector>

using namespace std;

class Solution{
    public:
        string longestPalindrome(string s){
            int n = s.size();
            if{n == 0} return s;

            int start = 0;
            int max_len = 1;
            vector<vector<bool>> dp(n, vector<bool>(n, false));

            // base case
            for(int i = 0; i < n; ++i)
                dp[i][i] = true;

            for(int i = 2; i < n; ++i){ // length of substring
                for(int j = 0; j < ; ++j){
                    
                }
            }
        }
};