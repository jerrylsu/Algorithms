#include<stack>

using std::stack;

/*
这是一个典型的栈的应用类问题
我们用一个栈来存储坐标。为了方便计算，在最开始的时候，我们将栈里面放入一个-1.
当遇到的是'('的时候，我们将其坐标入栈，
当遇到的是'）'的时候，弹出栈顶元素。此时需要分两种情况。
此时如果栈空了，其实相当于前面已经正好匹配了，然后再进来了一个'）',此时无需更新最大值max，
只需将当期坐标入栈。其作用和上面栈初始化的时候放入一个-1相同。
如果此时栈非空，说明又多了一对匹配。需要更新max的值。
*/

class Solution {
public:
    int longestValidParentheses(string s) {
        stack<int> stack;
        int max = 0;
        stack.push(-1);
        for(int i = 0; i < s.size(); i++){
            if(s[i] == '('){
                stack.push(i);
            }else{
                stack.pop();
                if(stack.empty()){
                    stack.push(i);
                }else{
                    max = max > i - stack.top() ? max : i - stack.top();
                }
            }
        }
        return max;
    }
};