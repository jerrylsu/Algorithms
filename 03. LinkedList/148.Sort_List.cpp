class ListNode {
    int val;
    ListNode* next;

    ListNode() : val(0), next(nullptr) { }
    ListNode(int val) : val(val), next(nullptr) { }
    ListNode(int val, ListNode* next) : val(val), next(next) { }
};


class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) { return head; }
        ListNode* fast = head;
        ListNode* slow = head;
        while (fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        fast = slow->next;
        slow->next = nullptr;
        return mergeTwoLists(sortList(head), sortList(fast));
    }

    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (l1 == nullptr && l1 ==nullptr) { return nullptr; }
        if (l1 == nullptr || l2 == nullptr) { return l1 == nullptr ? l2 : l1;}

        ListNode dummy(-1), *current;
        current = &dummy;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                current->next = l1;
                l1 = l1->next;
            }
            else{
                current->next = l2;
                l2 = l2->next;
            }
            current = current->next;
        }
        current->next = l1 == nullptr ? l2 : l1;
        return dummy.next;
    }
};
