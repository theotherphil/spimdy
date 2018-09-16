
use std::{
    arch::x86_64::*,
    ops::*
};

use super::{
    vector_types::*,
    bool_array_from_m256i
};

/// The set of lanes which are currently active in some execution path
#[derive(Copy, Clone)]
pub struct Mask {
    pub mask: __m256i
}

impl Mask {
    pub fn all_on() -> Mask {
        unsafe { Mask { mask: _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()) } }
    }

    pub fn all_off() -> Mask {
        unsafe { Mask { mask: _mm256_setzero_si256() } }
    }

    pub fn new(b: Vbool) -> Mask {
        Mask { mask: b.value }
    }

    pub fn any(&self) -> bool {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.mask)) != 0 }
    }

    pub fn andnot(&self, other: &Self) -> Self {
        unsafe { Mask { mask: _mm256_andnot_si256(self.mask, other.mask) } }
    }

    pub fn array(&self) -> [bool; 8] {
        bool_array_from_m256i(self.mask)
    }
}

impl_bin_op!(Mask, mask, BitAnd, bitand, _mm256_and_si256);
impl_bin_op!(Mask, mask, BitOr, bitor, _mm256_or_si256);
impl_bin_op!(Mask, mask, BitXor, bitxor, _mm256_xor_si256);

impl BitAndAssign for Mask {
    fn bitand_assign(&mut self, rhs: Self) {
        unsafe { self.mask = _mm256_and_si256(self.mask, rhs.mask); }
    }
}

impl_debug_for_m256i_wrapper!(Mask, mask, |a| a != 0);

pub struct Kernel {
    // Top level kernel mask
    pub kernel_exec: Mask,
    // Internal control flow mask
    pub internal_exec: Mask,
    // OR of all lanes which hit a continue in the current loop
    pub continue_exec: Mask, 
    // internal_mask & kernel_exec
    pub exec: Mask,
}

/// Creates an all-one kernel to start a chain of SPMD calculations.
/// Currently pretty pointless. I'm hoping to use some macro magic to
/// make SPMD calculations less verbose, but I've yet to work out how.
#[macro_export]
macro_rules! spmd {
    ($body:expr) => {
        let mut k = Kernel::new();
        $body(&mut k);
    };
}

/// We can't have an if-else function on Kernel taking if and else closures because the two closures typically
/// mutate a common set of variables, and the borrow checker doesn't like that.
#[macro_export]
macro_rules! if_else {
    ($k:ident, $cond:expr, $if_body:expr, $else_body:expr) => {
        $k.vif($cond, $if_body);
        $k.vif(!$cond, $else_body);
    };
}

/// As with if-else statements, we need to use a macro for this as the input closures often mutate some common
/// variables and the borrow checker wouldn't allow that if this were a function on Kernel.
#[macro_export]
macro_rules! vfor {
    ($k:ident, init: $for_init_body:expr, cond: $for_cond_body:expr, incr: $for_incr_body:expr, body: $for_body:expr) => {
        let old_internal_exec = $k.internal_exec;

        with_kernel($k, $for_init_body);

        let old_continue_exec = $k.continue_exec;
        $k.continue_exec = Mask::all_off();
  
        loop {
            let cond = with_kernel_ret($k, $for_cond_body);
            let cond_exec = Mask::new(cond);
            $k.internal_exec = $k.internal_exec & cond_exec;
            $k.exec = $k.exec & cond_exec;
      
            if !$k.exec.any() {
                break;
            }

            with_kernel($k, $for_body);

            // reactivate all lanes that hit a "continue"
            $k.internal_exec = $k.internal_exec | $k.continue_exec;
            $k.exec = $k.internal_exec & $k.kernel_exec;
            $k.continue_exec = Mask::all_off();

            with_kernel($k, $for_incr_body);
        }

        // now that the loop is done,
        // can restore lanes that were "break"'d, or that failed the loop condition.
        $k.internal_exec = old_internal_exec;
        $k.exec = $k.internal_exec & $k.kernel_exec;
        // restore the continue mask of the previous loop in the stack
        $k.continue_exec = old_continue_exec;
    }
}

/// A type that supports masked assignment, i.e. assigning values for all lanes which are on in the mask.
pub trait MaskedStore {
    fn masked_store(&mut self, mask: &Mask, value: Self);
}

impl MaskedStore for Vf32 {
    fn masked_store(&mut self, mask: &Mask, value: Self) {
        unsafe {
            let c = _mm256_castsi256_ps(mask.mask);
            let b = _mm256_blendv_ps(self.value, value.value, c);
            self.value = b;
        }
    }
}

impl MaskedStore for Vi32 {
    fn masked_store(&mut self, mask: &Mask, value: Self) {
        unsafe {
            let blended = _mm256_blendv_ps(
                _mm256_castsi256_ps(self.value),
                _mm256_castsi256_ps(value.value),
                _mm256_castsi256_ps(mask.mask)
            );
            self.value = _mm256_castps_si256(blended);
        }
    }
}

/// Stupid hack for type inference
pub fn with_kernel(k: &mut Kernel, mut f: impl FnMut(&mut Kernel)) {
    f(k);
}

/// Stupid hack for type inference
pub fn with_kernel_ret<T>(k: &mut Kernel, mut f: impl FnMut(&mut Kernel) -> T) -> T {
    f(k)
}

const PROGRAM_COUNT: i32 = 8;

// Make this a lazy_static
pub fn program_index() -> LinearInt {
    unsafe { 
        LinearInt { 
            value: _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)
        }
    }
}


impl Kernel {
    // Create a new kernel with this kernel's execution mask and default
    // values for the other masks
    pub fn clone_exec_mask(&self) -> Kernel {
        Self::masked(self.exec)
    }

    pub fn new() -> Kernel {
        Kernel { 
            kernel_exec: Mask::all_on(),
            internal_exec: Mask::all_on(),
            continue_exec: Mask::all_off(),
            exec: Mask::all_on()
        }
    }

    pub fn masked(kernel_exec: Mask) -> Kernel {
        Kernel { 
            kernel_exec: kernel_exec,
            internal_exec: Mask::all_on(),
            continue_exec: Mask::all_off(),
            exec: kernel_exec
        }
    }

    /// Use the if_else macro for if-else statements - see comment on that macro for why
    pub fn vif(&mut self, cond: Vbool, body: impl FnOnce(&mut Self)) {
        //println!("RUNNING VIF");

        let old_internal_exec = self.internal_exec;
        let cond_exec = Mask::new(cond);
        let pre_if_internal_exec = self.internal_exec & cond_exec;

        //println!("    INITIAL internal_exec: {:?}", self.internal_exec);
        //println!("    INITIAL exec: {:?}", self.exec);
        //println!("    INITIAL continue_exec: {:?}", self.continue_exec);

        self.internal_exec = self.internal_exec & cond_exec;
        self.exec = self.exec & cond_exec;

        //println!("    PRE internal_exec: {:?}", self.internal_exec);
        //println!("    PRE exec: {:?}", self.exec);
        //println!("    PRE continue_exec: {:?}", self.continue_exec);

        if self.exec.any() {
            body(self);
        }

        //println!("    POST internal_exec: {:?}", self.internal_exec);
        //println!("    POST exec: {:?}", self.exec);
        //println!("    POST continue_exec: {:?}", self.continue_exec);

        // propagate any lanes that were shut down inside the if
        self.internal_exec = (pre_if_internal_exec ^ self.internal_exec).andnot(&old_internal_exec);
        self.exec = self.kernel_exec & self.internal_exec;

        //println!("    FINALLY internal_exec: {:?}", self.internal_exec);
        //println!("    FINALLY exec: {:?}", self.exec);
        //println!("    FINALLY continue_exec: {:?}", self.continue_exec);
    }

    pub fn store<V: MaskedStore>(&self, dst: &mut V, value: V) {
        dst.masked_store(&self.exec, value);
    }

    pub fn foreach(&mut self, first: i32, last: i32, mut body: impl FnMut(&mut Self, LinearInt)) {
        assert!(first <= last);

        //println!("RUNNING FOREACH");

        let old_internal_exec = self.internal_exec;
        let old_continue_exec = self.continue_exec;
        self.continue_exec = Mask::all_off();
        let num_full_simd_loops = (last - first) / PROGRAM_COUNT;
        let num_partial_loops = (last - first) % PROGRAM_COUNT;

        let mut loop_index = first + program_index();

        //println!("FIRST: {:?}", first);
        //println!("PROGRAM_INDEX: {:?}", program_index());
        //println!("LOOP_INDEX: {:?}", loop_index);

        //println!("    FULL LOOPS: {}", num_full_simd_loops);

        for _ in 0..num_full_simd_loops {
            if !self.exec.any() {
                //println!("    EXEC ALL ZEROES");
                break;
            }

            //println!("    RUNNING BODY");
            body(self, loop_index);

            //println!("    PRE internal_exec: {:?}", self.internal_exec);
            //println!("    PRE exec: {:?}", self.exec);
            //println!("    PRE continue_exec: {:?}", self.continue_exec);

            // reactivate all lanes that hit a "continue"
            self.internal_exec = self.internal_exec | self.continue_exec;
            self.exec = self.internal_exec & self.kernel_exec;
            self.continue_exec = Mask::all_off();

            //println!("    POST internal_exec: {:?}", self.internal_exec);
            //println!("    POST exec: {:?}", self.exec);
            //println!("    POST continue_exec: {:?}", self.continue_exec);

            loop_index = loop_index + PROGRAM_COUNT;
        }

        if num_partial_loops > 0 && self.exec.any() {
            // apply mask for partial loop
            let partial_loop_mask = unsafe { 
                Mask {
                    mask:  _mm256_cmpgt_epi32(_mm256_set1_epi32(num_partial_loops), program_index().value)
                }
            };

            self.internal_exec = self.internal_exec & partial_loop_mask;
            self.exec = self.exec & partial_loop_mask;

            body(self, loop_index);
        }

        // now that the loop is done,
        // can restore lanes that were "break"'d, or that failed the loop condition.
        self.internal_exec = old_internal_exec;
        self.exec = self.internal_exec & self.kernel_exec;

        // restore the continue mask of the previous loop in the stack
        self.continue_exec = old_continue_exec;
    }

    pub fn vbreak(&mut self) {
        // this should only be called inside loops. A check for this would be good.
        // turn off all active lanes so nothing happens after the break.
        // this state will be restored after the current loop is done
        self.internal_exec = Mask::all_off();
        self.exec = Mask::all_off();
    }

    pub fn unmasked(&mut self, mut unmasked_body: impl FnMut(&mut Self)) {
        // beware: since we don't immediately return when all lanes turn off, zombie execution is possible.
        // save old exec mask
        let old_exec = self.exec;
        let old_kernel_exec = self.kernel_exec;
        let old_internal_exec = self.internal_exec;

        // totally unmask the execution
        self.kernel_exec = Mask::all_on();
        self.internal_exec = Mask::all_on();
        self.exec = Mask::all_on();

        // run the unmasked body
        unmasked_body(self);

        // restore execution mask with any new masks applied
        self.kernel_exec = self.kernel_exec & old_kernel_exec;
        self.internal_exec = self.internal_exec & old_internal_exec;
        self.exec = self.exec & old_exec;
    }

    // gather
    pub fn load_ref(&self, src: &Vf32LinearRef) -> Vf32 {
        unsafe {
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(self.exec.mask));
            if mask == 0b11111111 {
                // "all on" optimization: vector load
                Vf32 { value: _mm256_loadu_ps(src.value) }
            } else {
                // masked load
                Vf32 { value: _mm256_maskload_ps(src.value, self.exec.mask) }
            }            
        }
    }

    // scatter
    pub fn store_ref(&self, dst: &mut Vf32LinearRef, src: &Vf32) {
        unsafe {
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(self.exec.mask));
            if mask == 0b11111111 {
                // "all on" optimization: vector store
                _mm256_storeu_ps(dst.value, src.value);
            } else {
                // masked store
                _mm256_maskstore_ps(dst.value, self.exec.mask, src.value);
            }
        }
    }

     // gather
    pub fn load_ref_vi32(&self, src: &Vi32LinearRef) -> Vi32 {
        unsafe {
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(self.exec.mask));
            if mask == 0b11111111 {
                // "all on" optimization: vector load
                Vi32 { value: _mm256_loadu_si256(src.value as *mut __m256i) }
            } else {
                // masked load
                Vi32 { value: _mm256_maskload_epi32(src.value, self.exec.mask) }
            }            
        }
    }

    // scatter
    pub fn store_ref_vi32(&self, dst: &mut Vi32LinearRef, src: &Vi32) {
        unsafe {
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(self.exec.mask));
            if mask == 0b11111111 {
                // "all on" optimization: vector store
                _mm256_storeu_si256(dst.value as *mut __m256i, src.value);
            } else {
                // masked store
                _mm256_maskstore_epi32(dst.value, self.exec.mask, src.value);
            }
        }
    }
}
