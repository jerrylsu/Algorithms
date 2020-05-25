#include<vector>

using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val) : val(val), left(nullptr), right(nullptr) {}
};


class Solution{
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum){
        vector<vector<int>> res;
        vector<int> track;
        int count = 0;
        if(root == nullptr) return res;
        dfs(res, track, count, root, sum);
        return res;
    }

    void dfs(vector<vector<int>>& res, 
             vector<int>& track, 
             int count, 
             TreeNode* root, 
             int sum){
        if(root == nullptr)
            return;
        count += root->val;
        track.push_back(root->val);
        if(root->left == nullptr && root->right == nullptr) {
            if(count == sum)
                res.push_back(track);
        }
        dfs(res, track, count, root->left, sum);
        dfs(res, track, count, root->right, sum);
        count -= root->val;
        track.pop_back();
    }
};
