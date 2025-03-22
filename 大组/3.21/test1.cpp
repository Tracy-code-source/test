// 后缀表达式
// 未引入特殊字符区分，仅用个位数计算
#include <bits/stdc++.h>
using namespace std;
#define int long long
stack <int> D;
string num;
int ans;
signed main()
{
    cin >> num;
    for (int i = 0; num[i] != '\0'; i ++)
    {
        if (num[i] >= '0' && num[i] <= '9') D.push(num[i] - '0'); // 数字直接入栈
        else // 遇到运算符，弹出栈顶两个数计算再将结果入栈
        {
            int a = D.top();
            D.pop();
            int b = D.top();
            D.pop();
            if (num[i] == '+') D.push(a + b);
            else if (num[i] == '-') D.push(b - a); //注意减法除法顺序
            else if (num[i] == '*') D.push(a * b);
            else if (num[i] == '/') D.push(b / a);
        }
    }
    ans = D.top(); //运算结果
    cout << ans << endl;
    system("pause");
    return 0;
}