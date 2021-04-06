#include<iostream>
#include<vector>

using namespace std;

class Solution {
    public:
        // 暴力解法
        int maxProfit1(vector<int>& prices){
            int max_profit = 0;
            for(int i = 0; i < prices.size(); ++i){
                for(int j = i + 1; j < prices.size(); ++j){
                    max_profit = max(max_profit, prices[j] - prices[i]);
                }
            }
            return max_profit;
        }

        // DP
        int maxProfit(vector<int>& prices){
            int buy = prices[0];
            int max_profit = 0;
            for(auto& price : prices){
                buy = min(buy, price);
                max_profit = max(max_profit, price - buy);
            }
            return max_profit;
        }
};

int main(){
    vector<int> prices {7, 1, 5, 3, 6, 4};
    int max_profit;
    Solution solution;
    max_profit = solution.maxProfit(prices);
    cout << max_profit << endl;
    return 0;
}
