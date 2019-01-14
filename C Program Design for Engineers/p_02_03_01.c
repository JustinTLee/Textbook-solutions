/* C Program Design for Engineers, 2nd edition */
/* 2.3 - Programming Exercise: 1 */
/*
Write a statement that asks the user to type three integers and
another statement that stores the three user responses into first,
second, and third.
*/

#include <stdio.h>

int main(void)
{
  int first, second, third;

  printf("Enter the first integer: ");
  scanf("%d", &first);

  printf("Enter the second integer: ");
  scanf("%d", &second);

  printf("Enter the third integer: ");
  scanf("%d", &third);

  printf("First integer: %d \n", first);
  printf("Second integer: %d \n", second);
  printf("Third integer: %d \n", third);

  return(0):
}