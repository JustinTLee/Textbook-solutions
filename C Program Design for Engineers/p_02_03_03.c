/* C Program Design for Engineers, 2nd edition */
/* 2.3 - Programming Exercise: 3 */
/*
Write a program that tells a user the smallest tube of caulk that will
be big enough to seal around the user's windows and doors. Your
program should ask the user to enter how many centimeters of door and
window rims need caulking and how thick a bead (millimeters) will be
used. Note that a caulking bead is a cylinder of the given diameter.
Assume that you will need an additional 10% of the amount of caulk
calculated to allow some to be left in the tube. Tell the user the
volume (in cm^3) of the smallest tube with sufficient caulk. Be sure
to include a constant macro that defines PI as 3.14159.
*/

#include <stdio.h>
#include <math.h>
#define PI 3.14159
#define mm_to_cm 0.1

int main(void)
{
  double caulk_length, caulk_bead, caulk_vol;
  
  printf("How many centimeters of door and \
window rims need caulking? ");
  scanf("%lf", &caulk_length);

  printf("How thick (in mm) is a bead of caulk? ");
  scanf("%lf", &caulk_bead);

  caulk_vol = caulk_bead*mm_to_cm*PI*caulk_length*1.1;

  printf("You need %lf cm^3 of caulk to finish this.\n",
  	caulk_vol);
}