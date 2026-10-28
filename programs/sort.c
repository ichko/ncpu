// bubble sort of 4 numbers
int a[4] = {7, -3, 12, 0};
int i = 0, j = 0, t = 0;
for (i = 0; i < 3; i++) {
    for (j = 0; j < 3 - i; j++) {
        if (a[j] > a[j + 1]) { t = a[j]; a[j] = a[j + 1]; a[j + 1] = t; }
    }
}
