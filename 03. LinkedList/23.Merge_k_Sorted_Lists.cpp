#include<iostream>
#include<vector>

using namespace std;

class ListNode {
    int val;
    ListNode* next;
    
    ListNode() : val(0), next(nullptr) { }
    ListNode(int val) : val(val), next(nullptr) { }
    ListNode(int val, ListNode* next) : val(val), next(next) { }
};


class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
    /*O(K*N)*/
        if (lists.size() == 0) { return nullptr; }
        if (lists.size() == 1) { return lists[0]; }

        ListNode* res = lists[0];
        int n = lists.size();
        for (int i = 1; i < n; ++i) {
            res = mergeLists(res, lists[i]);
        }
        return res;
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
    /* 时间复杂度分析：K 条链表的总结点数是 N，平均每条链表有 N/K 个
     * 节点，因此合并两条链表的时间复杂度是 O(N/K)。从 K 条链表开始
     * 两两合并成 1 条链表，因此每条链表都会被合并 logK 次，因此 K 
     * 条链表会被合并 K * logK 次，因此总共的时间复杂度是 K*logK*N/K 
     * 即 O（NlogK）。*/
        return partition(lists, 0, lists.size() - 1);
    }

    ListNode* partition(vector<ListNode*>& lists, int start, int end) {
        if (start > end) {{ return nullptr; }
        if (start == end) { return lists[start]; }

        if (start < end) {
            int mid = start + ((end - start) >> 1);
            ListNode* l = partition(lists, start, mid);
            ListNode* r = partition(lists, mid + 1, end);
            return mergeTwoLists(l, r);
        }

        return nullptr;
    }

    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2){
        if (l1 == nullptr && l2 == nullptr) { return nullptr; }
        if (l1 == nullptr) { return l2; }
        if (l2 == nullptr) { return l1; }

        ListNode dummy(-1), *cur;
        cur = &dummy;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                cur->next = l1;
                l1 = l1->next;
            }
            else{
                cur->next = l2;
                l2 = l2->next;
            }
            cur = cur->next;
        }
        cur->next = l1 == nullptr ? l2 : l1;
        return dummy.next;
    }
};
