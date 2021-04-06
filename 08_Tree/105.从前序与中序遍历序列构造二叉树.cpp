#include<vector>
#include<unordered_map>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val) : val(val), left(nullptr), right(nullptr) {}
};


class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        unordered_map<int, int> map;
        int n = preorder.size();

        // 使用hash结构快速在中序遍历的结果中定位根结点的位置
        for(int i = 0; i < n; ++i) {
            map[inorder[i]] = i;
        }

        return myBuildTree(map, preorder, inorder, 0, n - 1, 0, n - 1);
    }

    TreeNode* myBuildTree(unordered_map<int, int>& map,
                        vector<int>& preorder,
                        vector<int>& inorder,
                        int preorder_left, 
                        int preorder_right, 
                        int inorder_left, 
                        int inorder_right) {
        if(preorder_left > preorder_right) return nullptr;

        // 前序遍历第一个结点为根结点
        int preorder_root = preorder_left;
        // 建立根结点
        TreeNode* root = new TreeNode(preorder[preorder_root]);

        // 根结点在中序中的位置
        int inorder_root = map[preorder[preorder_root]];

        // 中序的左子树左右区间
        int inorder_sublefttree_left = inorder_left;
        int inorder_sublefttree_right = inorder_root - 1;
        // 中序的右子树左右区间
        int inorder_subrighttree_left = inorder_root + 1;
        int inorder_subrighttree_right = inorder_right;

        // 前序的左子树左右区间
        int preorder_sublefttree_left = preorder_left + 1;
        int preorder_sublefttree_right = inorder_sublefttree_right - inorder_sublefttree_left + preorder_sublefttree_left;
        // 前序的右子树左右区间
        int preorder_subrighttree_left = preorder_sublefttree_right + 1;
        int preorder_subrighttree_right = preorder_right;

        root->left = myBuildTree(map, preorder, inorder,
                                 preorder_sublefttree_left, preorder_sublefttree_right,
                                 inorder_sublefttree_left, inorder_sublefttree_right);
        root->right = myBuildTree(map, preorder, inorder,
                                  preorder_subrighttree_left, preorder_subrighttree_right,
                                  inorder_subrighttree_left, inorder_subrighttree_right);

        return root;

    }
};
