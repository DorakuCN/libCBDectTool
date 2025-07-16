int main(int argc, char* argv[]) {
  printf("Testing our image...\n");
  detect("example_data/04.png", cbdetect::SaddlePoint);
  return 0;
} 