#include<algorithm>

class TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val) : val(val), left(nullptr), right(nullptr {}
};


class Solution {
public:
    bool isSymmetrical(TreeNode* root){
        if(root == nullptr) return true;
        return helper(root->left, root->right);
    }

    bool helper(TreeNode* node1, TreeNode* node2){
        if(node1 == nullptr && node2 == nullptr) return true;
        if(node1 == nullptr || node2 == nullptr) return false;
        if(node1->val != node2->val) return false;
        return helper(node1->left, node2->right) && helper(node1->right, node2->left);
    }
};
