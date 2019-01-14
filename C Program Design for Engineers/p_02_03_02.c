/* C Program Design for Engineers, 2nd edition */
/* 2.3 - Programming Exercise: 2 */
/*
a. Write a statement that dispalys the following line with the value
of the type double variable x at the end.

The value of x is ______.

b. Assuming radius and surface are type double variables containing
the radius and surface area of a sphere, write a statement that will
display this information in the form:

The surface area of a sphere with radius ______ is ______.
*/

#include <stdio.h>
#include <math.h>
#define PI 3.14159

int main(void)
{
  double x, radius, surface;

  printf("Enter a value for x: ");
  scanf("%lf", &x);

  printf("The value of x is %1f. \n", x);

  printf("Enter a value for the radius: ");
  scanf("%lf", &radius);

  surface = 4 * PI * pow(radius, 2.0);
	
  printf("The surface are of a sphere with radius %lf is %lf. \n",
  	radius, surface);

  return(0);
}