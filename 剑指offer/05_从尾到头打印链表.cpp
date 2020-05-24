#include<iostream>
#include<vector>
#include<stack>

class ListNode {
public:
    int val;
    ListNode* next; 
    ListNode() : val(0), next(nullptr) { }
    ListNode(int val) : val(val), next(nullptr) { }
    ListNode(int val, ListNode* next) : val(val), next(next) { }
};

class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        vector<int> res;
        helper(res, head);
        return res;
    }

    void helper(vector<int>& res, ListNode* head) {
        if(head == nullptr) return;
        helper(res, head->next);
        res.push_back(head->val);
    }

    vector<int> reversePrint2(ListNode* head) {
        stack<int> stack;
        vector<int> res;

        while(head != nullptr) {
            stack.push(head->val);
            head = head->next;
        }

        while(!stack.empty()) {
            res.push_back(stack.top());
            stack.pop();
        }

        return res;
    }
};
