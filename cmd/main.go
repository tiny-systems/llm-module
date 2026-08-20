package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/rs/zerolog"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	_ "github.com/tiny-systems/llm-module/components/llmchat"
	_ "github.com/tiny-systems/llm-module/components/llmcomplete"
	_ "github.com/tiny-systems/llm-module/components/llmrouter"
	_ "github.com/tiny-systems/llm-module/components/llmtoolresult"
	_ "github.com/tiny-systems/llm-module/components/llmtools"
	_ "github.com/tiny-systems/llm-module/components/mcpcall"
	_ "github.com/tiny-systems/llm-module/components/mcptools"
	"github.com/tiny-systems/module/cli"
)

var rootCmd = &cobra.Command{
	Use:   "server",
	Short: "Tiny Systems LLM module — chat, completion, tool-calling, routing, and MCP client components",
	Run: func(cmd *cobra.Command, args []string) {
		_ = cmd.Help()
	},
}

func main() {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	viper.AutomaticEnv()
	if viper.GetBool("debug") {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cli.RegisterCommands(rootCmd)
	if err := rootCmd.ExecuteContext(ctx); err != nil {
		fmt.Printf("command execute error: %v\n", err)
	}
}
