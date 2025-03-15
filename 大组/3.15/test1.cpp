// 单项链表ADT
#include <bits/stdc++.h>
using namespace std;
#define int long long
struct Info
{
    int data; // 有效数据
    struct Info* next; // 指向节点的指针
};
struct Info* Create()
{
    struct Info* head = NULL; // 头指针
    struct Info* current, *prev; // 流动指针, 前节点指针
    current = prev = (struct Info*)malloc(sizeof(struct Info));
    if (head == NULL) //头指针不包含有效值， 方便后续统一操作
    {
        head = (struct Info*)malloc(sizeof(struct Info));
        head->next = current;
        prev = head;
    }
    cin >> current->data;
    while (current->data != 0) //以数据为零作为循环终点
    {
        prev->next = current;
        prev = current;
        current = (struct Info*)malloc(sizeof(struct Info)); // 重新为流动指针分配空间
        cin >> current->data;
    }
    prev->next = NULL; // 尾指针指向空
    return head; //返回头指针
}
signed main()
{
    struct Info* pre; // 声明一个结构体指针
    pre = Create(); // 将头指针赋给新指针，便于维护
    pre = pre->next;
    while (pre != NULL)// 遍历检验是否建立成功
    {
        cout << pre->data << ' ';
        pre = pre->next;
    }
    system("pause");
    return 0;
}
