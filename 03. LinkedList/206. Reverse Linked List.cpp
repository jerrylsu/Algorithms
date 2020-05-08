#include<iostream>

using namespace std;

class ListNode{
public:
    int val;
    ListNode* next;

    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next)  : val(x), next(next) {}

};


class Solution{
public:
    ListNode* reverseList(ListNode* head){
        if(!head) return nullptr;
	ListNode *pre, *cur, *post;
	cur = head;
	pre = nullptr;
	while(cur){
	    post = cur->next;
	    cur->next = pre;
	    pre = cur;
	    cur = post;
	}
	return pre;
    }
};
