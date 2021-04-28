class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        
class Solution:
    def reverseLinkedList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        pre, cur, post = None, head, head.next
        while cur:
            post = cur.next
            cur.next = pre
            pre = cur
            cur = post
        return pre
