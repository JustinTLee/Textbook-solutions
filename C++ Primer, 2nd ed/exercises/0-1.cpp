#include <iostream>
using namespace std;

main()
{
    char ch;
    int lineCnt = 0, charCnt = 0;

    while (cin.get(ch))
    {
        // break statements cause program to execute next lines after switch statement
        switch(ch)
        {
            // does the lack of anything in the tab case move to the default case?
            // JL - seems like it:
            //     adding a test case as: 1234    123 prints out a count of 7.
            //     so then why use the break statements?
            //     so not having a break statement there is causing the switch statement to move tab to the case ' ', and the break statement there will apply
            case '\t':
            case ' ':
                break;
            case '\n':
                ++lineCnt;
                break;
            default:
                ++charCnt;
                break;
        }
    }

    cout << "Line count: " << lineCnt << endl;
    cout << "Char count: " << charCnt << endl;
    return 0;
}