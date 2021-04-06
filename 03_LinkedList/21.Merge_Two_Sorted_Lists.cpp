#include<iostream>

using namespace std;

class ListNode{
    public:
        int val;
        ListNode* next;
    
	ListNode() : val(0), next(nullptr) {}
	ListNode(int x) : val(x), next(nullptr) {}
	ListNode(int x, ListNode* next) : val(x), next(next) {}
};


class Solution{
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2){
	    if(!l1 && !l2) return nullptr;
	    if(!l1 || !l2) return !l1 ? l2 : l1;

	    ListNode dummy(-1), *current;
	    current = &dummy;
	    while(l1 && l2){
	        if(l1->val < l2->val){
	            current->next = l1;
		    l1 = l1->next;
	        }else{
	            current->next = l2;
		    l2 = l2->next;
	        }
	        current = current->next;
	    }
	    current->next = l1 == nullptr ? l2 : l1;
	    return dummy.next;
    }

    void printLinkedList(ListNode* l){
        while(l){
	    std::cout << l->val << " ";
	    l = l->next;
	}
	std::cout << std::endl;
    }
};


int main(){
    ListNode l1(1), n1(4), n2(6), l2(2), n3(5), n4(7);

    l1.next = &n1;
    n1.next = &n2;
    l2.next = &n3;
    n3.next = &n4;

    Solution solution;
    solution.printLinkedList(&l1);
    solution.printLinkedList(&l2);
    solution.mergeTwoLists(&l1, &l2);
    return 0;
}
