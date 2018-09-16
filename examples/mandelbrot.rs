
#[macro_use]
extern crate spimdy;
extern crate image;

use spimdy::*;
use spimdy::examples::*;
use image::*;
use std::time::Instant;

fn main() {
       // width needs to be evenly divisible by 8
        let width = 768;
        let height = 512;
        let x0 = -2.0;
        let x1 = 1.0;
        let y0 = -1.0;
        let y1 = 1.0;
        let max_iterations = 255;
        let mut out1 = vec![0; width * height];
        let mut out2 = vec![0; width * height];

        // This shows the vec as having the correct allocation for a __m256 (by accident...)
        // unsafe {
        //     let (a, b, c) = out1.align_to::<__m256>();
        //     println!("{}", a.len());
        //     println!("{}", b.len());
        //     println!("{}", c.len());
        // }

        let single_start = Instant::now();
        mandelbrot_single(x0, y0, x1, y1, width as i32, height as i32, max_iterations, &mut out1);
        let single_end = Instant::now();
        println!("SINGLE: {:?}", single_end - single_start);

        let spmd_start = Instant::now();
        spmd!(|k| {
            mandelbrot_spmd(k, x0, y0, x1, y1, width as i32, height as i32, max_iterations, &mut out2)
        });
        let spmd_end = Instant::now();
        println!("SPMD: {:?}", spmd_end - spmd_start);

        let single_u8: Vec<u8> = out1.iter().map(|x| *x as u8).collect();
        let spmd_u8: Vec<u8> = out2.iter().map(|x| *x as u8).collect();

        let single_image = GrayImage::from_vec(width as u32, height as u32, single_u8).unwrap();
        let spmd_image = GrayImage::from_vec(width as u32, height as u32, spmd_u8).unwrap();

        single_image.save("mandelbrot_single.png").unwrap();
        spmd_image.save("mandelbrot_spmd.png").unwrap();
}