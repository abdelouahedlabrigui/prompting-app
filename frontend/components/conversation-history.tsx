"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Trash2, Clock, User, Bot, Search } from "lucide-react"
import { Input } from "@/components/ui/input"
import type { Message } from "@/app/page"
import { CodeBlock } from "./code-block"

interface ConversationHistoryProps {
  messages: Message[]
  onClearHistory: () => void
}

export function ConversationHistory({ messages, onClearHistory }: ConversationHistoryProps) {
  const [searchTerm, setSearchTerm] = useState("")

  const filteredMessages = messages.filter(
    (message) =>
      message.userPrompt.toLowerCase().includes(searchTerm.toLowerCase()) ||
      message.response.toLowerCase().includes(searchTerm.toLowerCase()) ||
      message.model.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
  }

  const formatResponseTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`
    return `${(ms / 1000).toFixed(1)}s`
  }

  return (
    <Card className="h-[calc(100vh-12rem)]">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">History</CardTitle>
          <Button variant="outline" size="sm" onClick={onClearHistory} disabled={messages.length === 0}>
            <Trash2 className="w-4 h-4 mr-2" />
            Clear
          </Button>
        </div>

        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Filter history..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <ScrollArea className="h-[calc(100vh-20rem)]">
          <div className="p-4 space-y-4">
            {filteredMessages.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                {messages.length === 0 ? (
                  <p>No conversations yet. Start by sending a message!</p>
                ) : (
                  <p>No conversations match your search.</p>
                )}
              </div>
            ) : (
              filteredMessages.map((message) => (
                <Card key={message.id} className="border-l-4 border-l-primary/30">
                  <CardContent className="p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <Badge variant="outline" className="text-xs">
                        {message.model}
                      </Badge>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Clock className="w-3 h-3" />
                        {formatTime(message.timestamp)}
                        <span>•</span>
                        {formatResponseTime(message.responseTime)}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="flex items-start gap-2">
                        <User className="w-4 h-4 mt-1 text-primary flex-shrink-0" />
                        <div className="flex-1">
                          <p className="text-sm font-medium mb-1">You</p>
                          <p className="text-sm text-muted-foreground break-words">{message.userPrompt}</p>
                        </div>
                      </div>

                      <div className="flex items-start gap-2">
                        <Bot className="w-4 h-4 mt-1 text-secondary flex-shrink-0" />
                        <div className="flex-1">
                          <p className="text-sm font-medium mb-1">
                            {message.model} (Length: {message.response.length})
                          </p>
                          <div className={`text-sm ${message.success ? "" : "text-destructive"}`}>
                            <div className="bg-muted p-2 rounded text-xs font-mono overflow-auto">
                              <CodeBlock content={message.response}/>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
