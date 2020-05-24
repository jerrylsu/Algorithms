#include<algorithm>
#include<stack>


class TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val) : val(val), left(nullptr), right(nullptr {}
};


class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root){
        if(root == nullptr) return nullptr;
        // TreeNode* tmp = root->left;
        // root->left = root->right;
        // root->right = tmp;
        std::swap(root_left, root->right);
        mirrorTree(root->left);
        mirrorTree(root->right);
        return root;
    }
};
