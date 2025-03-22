// 链栈
#include <bits/stdc++.h>
using namespace std;
#define int long long
struct Info
{
    int data; // 数据
    struct Info* next;
};
struct Info* stackCreate() //创建栈
{
    struct Info* top = NULL;
    return top;
}
struct Info* Push(struct Info* top) //入栈
{
    struct Info* newnode = (struct Info*)malloc(sizeof(struct Info));
    cin >> newnode->data;
    newnode->next = top; // 往回指向前一个
    top = newnode; // 更新栈顶
    return top;
}
struct Info* Pop(struct Info* top)
{
    if (top == NULL) return top;
    struct Info* temp = top;
    top = top->next;
    cout << temp->data << endl; // 显示弹出节点数据
    free(temp);
    return top;
}
void Show(struct Info* top)
{
    if (top == NULL) return;
    cout << top->data << endl;
}
signed main()
{
    struct Info* top = stackCreate();
    for (int i = 0; i < 3; i ++)
    {
        //入栈及显示
        top = Push(top); 
        Show(top);
    }
    for (int i = 0; i < 3; i ++)
    {
        top = Pop(top);
    }
    system("pause");
    return 0;
}