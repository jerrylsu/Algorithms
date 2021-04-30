class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        dummpy1 = odd = ListNode(-1)
        dummpy2 = even = ListNode(-1)
        i = 1
        while head:
            if i % 2:
                odd.next = head
                odd = odd.next
            else:
                even.next = head
                even = even.next
            head = head.next
            i += 1
        odd.next = dummpy2.next
        even.next = None
        return dummpy1.next
