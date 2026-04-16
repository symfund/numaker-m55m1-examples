#if defined(GPIO_INT_BUTTON)
#include "NuMicro.h"

volatile uint8_t button0Pressed = 0;
volatile uint8_t button1Pressed = 0;

NVT_ITCM void GPI_IRQHandler(void)
{
    volatile uint32_t temp;

    /* To check if PI.11 interrupt occurred */
    if (GPIO_GET_INT_FLAG(PI, BIT11))
    {
        GPIO_CLR_INT_FLAG(PI, BIT11);
        button0Pressed = 1;
    }
    else
    {
        /* Un-expected interrupt. Just clear all PB interrupts */
        temp = PI->INTSRC;
        PI->INTSRC = temp;
        button0Pressed = 0;
    }
}

NVT_ITCM void GPH_IRQHandler(void)
{
    volatile uint32_t temp;

    /* To check if PI.11 interrupt occurred */
    if (GPIO_GET_INT_FLAG(PH, BIT1))
    {
        GPIO_CLR_INT_FLAG(PH, BIT1);
        button1Pressed = 1;
    }
    else
    {
        /* Un-expected interrupt. Just clear all PH interrupts */
        temp = PH->INTSRC;
        PH->INTSRC = temp;
        button1Pressed = 0;
    }
}

void button_init(void)
{
    GPIO_SetMode(PI, BIT11, GPIO_MODE_INPUT);
    GPIO_SetPullCtl(PI, BIT11, GPIO_PUSEL_PULL_UP);

    GPIO_EnableInt(PI, 11, GPIO_INT_FALLING);
    NVIC_EnableIRQ(GPI_IRQn);

    GPIO_SetMode(PH, BIT1, GPIO_MODE_INPUT);
    GPIO_SetPullCtl(PH, BIT1, GPIO_PUSEL_PULL_UP);

    GPIO_EnableInt(PH, 1, GPIO_INT_FALLING);
    NVIC_EnableIRQ(GPH_IRQn);
}
#endif /* GPIO_INT_BUTTON */
